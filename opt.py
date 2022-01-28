#!/usr/bin/python3
# Optimise an ONIOM geometry using Python's scipy.optimize
# Can be used as a last resort if there are issues with Gaussian's optimiser
# Author: Tristan Mackenzie

import sys
import os
import subprocess
import numpy as np
import signal
from scipy import optimize

directory = "."
settings_file = ""

# Is this running on a local machine or HPC?
local = False

if local:
    HOME = "."
    WORK = "."
    G16 = '/PATH/TO/g16'
    LAST_GEO = "last_geo.sh"
else:    
    # Use these if input and output files are stored in different locations
    # Otherwise they can be the same directory
    HOME = "/PATH/TO/HOME/DIRECTORY/"
    WORK = "/PATH/TO/WORK/DIRECTORY/"
    G16 = 'g16'
    LAST_GEO = "last_geo.sh"

class Opt():

    def __init__(self, base_path, fname):

        self.initial_com_path   = os.path.join(HOME, base_path, fname + ".com")
        self.gradient_com_path  = os.path.join(HOME, base_path, fname + "_grad.com")
        self.log_path           = os.path.join(WORK, base_path, fname + ".log")
        self.host_path          = os.path.join(HOME, base_path, fname + ".host")

        self.gradient_updater = Updater(self.initial_com_path, self.gradient_com_path)
        self.positions = self.gradient_updater.positions_from_com_details()

        n = self.natoms = self.positions.shape[0]

        print(f"Initial com path: {self.initial_com_path}")
        print(f"Gradient com path: {self.gradient_com_path}")
        print(f"Output path: {self.log_path}")
        print(f"Num Atoms: {n}")

        self.stepnum=0
        self.numsteps=0

    def run(self, nsteps, method='BFGS'):

        self.stepnum = 1
        self.numsteps = nsteps

        positions = self.positions.flatten()

        print(optimize.minimize(
            fun=self.gradient, 
            x0=positions, 
            method=method, 
            jac=True, 
            options={'maxiter':nsteps}
        ))

    def gradient(self, positions):

        self.gradient_updater.update(positions.reshape(self.natoms, 3))
        return self.run_gaussian(self.gradient_com_path)

  
    def run_gaussian(self, input_file):

        command = [G16 + " < " + input_file]

        process = subprocess.Popen(
            command, 
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            shell=True
        )

        first_step = self.stepnum == 1
        
        flag = "wb" if first_step else "ab"
        fi = 0
        fskip = 0
        fread = False
        write_line = first_step

        energy = 0
        forces = np.zeros((self.natoms * 3))

        with open(self.log_path, flag) as fo: 
            for line in iter(process.stdout.readline, b''):

                if line.startswith(b" ONIOM: extrapolated"):
                    energy = float(line.split()[-1])
                if line.startswith(b" Center     Atomic             Integrated Forces (Hartrees/Bohr)"):
                    fread=True
                    fskip=2

                elif line.endswith(b"/l9999.exe)\n") and self.stepnum != self.numsteps:
                    write_line=False
                elif line.endswith(b"/l202.exe)\n"):
                    write_line=True
                elif line.startswith(b" Step number"):
                    fo.write(f" Step number {self.stepnum:3d} out of a maximum of {self.numsteps}\n".encode())
                    self.stepnum += 1
                    continue

                elif fread:
                    if fskip > 0:
                        fskip -= 1
                    else:

                        if fi == self.natoms:
                            fread = False
                        else:
                            forces[fi*3:fi*3+3] = np.array([-float(f) for f in line.split()[-3:]])
                            fi += 1

                if write_line:
                    fo.write(line)

        return (energy, forces)

class Updater(object):

    def __init__(self, old_com, new_com=""):

        if not os.path.exists(old_com) and not os.path.splitext(old_com)[1] in [".com", ".gjf"]:
            raise IOError("old_com must be a .com or .gjf file")

        self.old_com = old_com

        if new_com:
                self.new_com = new_com
        else:
            self.new_com = old_com

        self.read_com()

    def read_com(self):
        with open(self.old_com, "r") as com_obj:
            com_str = com_obj.read()

        c = {
                "pregeom"      : [],
                "atoms"        : [],
                "additional"   : []
            }

        phase = 0
        for l in com_str.split("\n"):
            if phase == 0:
                if len(l) > 2 and  l.strip()[0].isdigit() and l.strip()[1] == " ":
                    #Charge/Mult line
                    phase += 1
                c["pregeom"].append(l)
            elif phase == 1:
                if l:
                    c["atoms"].append(l.strip())
                else:
                    if not c["atoms"]:
                        raise RuntimeError("No atoms section in com")
                    phase += 1
            elif phase == 2:
                c["additional"].append(l)

        self.com_details = c
        self.positions = self.positions_from_com_details()

    def positions_from_com_details(self):
        c = self.com_details
        ps = []
        for a in c["atoms"]:
            sa = a.split()
            if sa[1] == "0":
                ps.append([float(p) for p in sa[2:5]])
            else:
                ps.append([float(p) for p in sa[1:4]])
        return np.array(ps)

    def write_com(self):
        c = self.com_details
        p = self.positions
        s = "\n".join(c["pregeom"])
        s += "\n"

        for i, a in enumerate(c["atoms"]):
            sa = a.split()
            if sa[1] in ["0", "-1"]:
                symbol_flag = " ".join(sa[:2])
                oniom_str = " ".join(sa[5:])
            else:
                symbol_flag = sa[0] + " 0"
                oniom_str = " ".join(sa[4:])
                
            s += symbol_flag + " {:-12.7f}{:-12.7f}{:-12.7f} ".format(p[i,0], p[i,1], p[i,2]) + oniom_str + "\n"
        s += "\n"

        s += "\n".join(c["additional"])

        with open(self.new_com, "w") as com_file:
            com_file.write(s)

    def get_keywords(self):
        for l in self.com_details["pregeom"]:
            if l.startswith("#"):
                return l

    def set_keywords(self, new_keywords):
        ln = -1
        for i, l in enumerate(self.com_details["pregeom"]):
            if l.startswith("#"):
                ln = i
                break

        if ln == -1:
            raise Exception("Failed to set keywords - keywords line not found in input file")

        self.com_details["pregeom"][ln] = new_keywords

    def update(self, positions):
        self.positions = positions
        self.write_com()

def default_sigpipe():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)


if __name__ == "__main__":
    args = sys.argv
    
    path = args[1]
    fname = args[2]
    steps = int(args[3])
    method=args[4] if len(args) > 4 else 'BFGS'

    opt = Opt(path, fname)
    opt.run(steps, method)

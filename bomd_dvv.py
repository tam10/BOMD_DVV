#!/usr/bin/python3
# Born-Oppenheimer Molecular Dynamics with Damped Velocity Verlet for ONIOM in Gaussian
# Atoms can be heavily damped if they move in an uphill direction or the angle between their velocity and force is large 
# Author: Tristan Mackenzie

import sys
import os
import subprocess
import numpy as np
import configparser
import signal

dt_default = 0.1
damp_default = 0.95
vi_mult_default = 1
uphill_damp_default = 0

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

class Dynamics():

    def __init__(self, base_path, fname):

        self.initial_com_path = os.path.join(HOME, base_path, fname + ".com")
        self.update_com_path  = os.path.join(HOME, base_path, fname + "_updated.com")
        self.log_path         = os.path.join(WORK, base_path, fname + ".log")
        self.config_path      = os.path.join(HOME, base_path, fname + ".ini")
        self.host_path        = os.path.join(HOME, base_path, fname + ".host")

        self.updater = Updater(self.initial_com_path, self.update_com_path)
        self.positions = self.updater.positions_from_com_details()

        n = self.natoms = self.positions.shape[0]

        print(f"Initial com path: {self.initial_com_path}")
        print(f"Updated com path: {self.update_com_path}")
        print(f"Output path: {self.log_path}")
        print(f"Config path: {self.config_path}")
        print(f"Num Atoms: {n}")

        self.forces = np.zeros((n, 3), dtype=float)
        self.velocities = np.zeros((n, 3), dtype=float)
        self.new_velocities = np.zeros((n, 3), dtype=float)
        self.accelerations = np.zeros((n, 3), dtype=float)
        self.new_accelerations = np.zeros((n, 3), dtype=float)
        self.masses = np.zeros((n),dtype=int)
        self.atomic_nums = np.zeros((n),dtype=int)

        self.config = None

        self.predamp = np.ones((n),dtype=float)
        self.dot_matrix = np.ones((n),dtype=float)

        self.ekin = 0
        self.ekin_damp = 0
        self.epot = 0
        self.etot = 0

        self.stepnum=0
        self.numsteps=0
        self.first_step = False
        self.last_step = False

    def write_host(self, string):
        with open(self.host_path, "a") as fo:
            fo.write(string)

    def run(self, nsteps):

        self.stepnum = 1
        self.numsteps = nsteps

        for _ in range(nsteps):
            self.first_step = self.stepnum == 1
            self.last_step = self.stepnum == self.numsteps

            print(f"Step: {self.stepnum}")
            self.step()
            self.stepnum+=1

    def step(self):

        # Get config
        self.config = self.read_config()
        dt = self.config['dt']
        damp = self.config['damp']
        vi_mult = self.config['vi_mult']
        uphill_damp = self.config['uphill_damp']

        self.run_gaussian()

        if not np.any(self.forces):
            print("No gradient")
            exit(1)

        for i in range(self.natoms):
            v = self.velocities[i]
            f = self.forces[i]

            # Get dot product between vectors
            d = np.dot(v, f)

            # Ignore zero vectors
            if d == 0:
                self.dot_matrix[i] = 1.
                self.predamp[i] = 1.
                continue

            # Normalise
            d /= (np.linalg.norm(v) * np.linalg.norm(f))
            self.dot_matrix[i] = d

            # Map dot from (-1)->(1) to (0)->(1) if uphill_damp == 1, or (1) if uphill_damp == 0 or anything in between
            self.predamp[i] = p = uphill_damp * 0.5 * (d + 1) + (1 - uphill_damp)

            self.velocities[i] *= p

        # Verlet
        self.positions += self.velocities * dt + self.accelerations * (dt**2) * 0.5
        self.new_accelerations = self.forces / self.masses[:,None] #Slice notation to do row-wise division
        self.new_velocities = self.velocities + (self.accelerations + self.new_accelerations) * (dt * 0.5)

        if self.first_step:
            self.new_velocities *= vi_mult
            
        self.ekin = 0.5 * np.sum((self.masses[:,None] * self.new_velocities).T.dot(self.new_velocities))
        if uphill_damp > 0:
            self.new_velocities = damp * self.new_velocities * (2 - self.predamp[:, None])
        else:
            self.new_velocities = damp * self.new_velocities

        self.ekin_damp = 0.5 * np.sum((self.masses[:,None] * self.new_velocities).T.dot(self.new_velocities))

        self.velocities = self.new_velocities
        self.accelerations = self.new_accelerations

        self.updater.update(self.positions)

        if self.first_step:
            keywords = self.updater.get_keywords()
            keywords += " guess=read"
            self.updater.set_keywords(keywords)
        
    def run_gaussian(self):

        input_file = self.initial_com_path if self.first_step else self.update_com_path

        command = [G16 + " < " + input_file]

        process = subprocess.Popen(
            command, 
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            shell=True
        )
        
        flag = "wb" if self.first_step else "ab"
        mi = 0
        fi = 0
        fskip = 0
        iskip = 2
        fread = False
        iwrite = False
        write_line = self.first_step

        with open(self.log_path, flag) as fo: 
            for line in iter(process.stdout.readline, b''):

                #line = bline.decode()

                if line.startswith(b" AtmWgt="):
                    for m in line.split()[1:]:
                        self.masses[mi] = float(m)
                        mi += 1

                elif line.startswith(b" Center     Atomic             Integrated Forces (Hartrees/Bohr)"):
                    fread=True
                    fskip=2

                elif line.endswith(b"/l9999.exe)\n") and not self.last_step:
                    write_line=False
                elif line.endswith(b"/l202.exe)\n"):
                    write_line=True
                elif line.startswith(b" Step number"):
                    fo.write(f" Step number {self.stepnum:3d} out of a maximum of {self.numsteps}\n".encode())
                    continue

                elif fread:
                    if fskip > 0:
                        fskip -= 1
                    else:

                        if fi == self.natoms:
                            fread = False
                            iwrite = True
                            iskip = 1
                        else:
                            self.forces[fi] = np.array([float(f) for f in line.split()[-3:]])
                            fi += 1
                elif iwrite:
                    if iskip > 0:
                        iskip -= 1
                    else:
                        fo.write(f" Config: {self.config}\n".encode())

                        max_a = 0.
                        sum_a = 0.

                        for d in self.dot_matrix:
                            if d < 1:
                                a = np.rad2deg(np.arccos(d))
                                if a > max_a:
                                    max_a = a
                                sum_a += a * a
                                
                        fo.write(f" Force/Velocity angles: Max={max_a:9.4f}. RMS={(sum_a / self.natoms) ** 0.5:9.4f}\n".encode())

                        for l in self.iter_velocities():
                            fo.write(l)
                        for l in self.iter_accelerations():
                            fo.write(l)
                        iwrite = False

                if write_line:
                    fo.write(line)

    def write_config(self):
        parser = configparser.ConfigParser()
        default_config = {
            'dt':dt_default,
            'damp':damp_default,
            'vi_mult':vi_mult_default,
            'uphill_damp':uphill_damp_default
        }
        parser['Settings'] = {k:str(v) for k,v in default_config.items()}
        with open(self.config_path, 'w') as configfile:
            parser.write(configfile)

        return default_config

    def read_config(self):
        if not os.path.exists(self.config_path):
            return self.write_config()

        else:
            parser = configparser.ConfigParser()
            parser.read(self.config_path)

            config = {}
            config['dt'] = parse(parser, 'dt', dt_default)
            config['damp'] = parse(parser, 'damp', damp_default)
            config['vi_mult'] = parse(parser, 'vi_mult', vi_mult_default)
            config['uphill_damp'] = parse(parser, 'uphill_damp', uphill_damp_default)

            return config

    def iter_velocities(self):
        yield b" -------------------------------------------------------------------\n"
        yield b" Center     Atomic                         Velocities               \n"
        yield b" Number     Number              X              Y              Z     \n"
        yield b" -------------------------------------------------------------------\n"

        for i in range(self.natoms):
            p = self.velocities[i]
            a = self.atomic_nums[i]
            yield f" {i+1:6d}     {a:4d}        {p[0]:14.9f} {p[1]:14.9f} {p[2]:14.9f}\n".encode()

        yield b" -------------------------------------------------------------------\n"
        yield f" Kinetic Energy: {self.ekin_damp:14.9f}\n\n".encode()

    def iter_accelerations(self):
        yield b" -------------------------------------------------------------------\n"
        yield b" Center     Atomic                       Accelerations              \n"
        yield b" Number     Number              X              Y              Z     \n"
        yield b" -------------------------------------------------------------------\n"

        for i in range(self.natoms):
            p = self.accelerations[i]
            a = self.atomic_nums[i]
            yield f" {i+1:6d}     {a:4d}        {p[0]:14.9f} {p[1]:14.9f} {p[2]:14.9f}\n".encode()

        yield b" -------------------------------------------------------------------\n\n"

def parse(parser, name, default):
    try:
        value = parser['Settings'][name]
        value = float(value)

    except KeyError:
        print(f"Item {name} not found in config file. Using default value {default} instead.")
        value = default
    except ValueError:
        print(f"Failed to parse '{value}' as float for {name}. Using default value {default} instead.")
        value = default
    return value

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

    dynamics = Dynamics(path, fname)
    dynamics.run(steps)



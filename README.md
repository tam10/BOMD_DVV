# BOMD_DVV
Born-Oppenheimer Molecular Dynamics with Damped Velocity Verlet for ONIOM in Gaussian

This script was made for problematic geometries that need to be optimised using the (TD) ONIOM(QM/MM)-EE scheme.
It avoids the use of the full Hessian.

Also bundled with:
* opt.py, which wraps around scipy.optimize - this can be more stable than Gaussian for the last steps.
* last_geo.sh - a bash script that pipes the last geometry in a Gaussian log file which is recommended for large files.

To use both, you need to supply an input FORCE calculation (replace 'opt' keyword with 'force') and call:
python3 bomd_dvv.py $fname $steps
python3 opt.py $fname $steps [$method]

bomd_dvv.py generates an ini file, that can be modified at runtime - this allows you to change parameters while the job is running. You can change:
* dt - timestep
* damp - velocity multiplier: 0=no velocity, 1=no damping
* vi_mult - initial velocity multiplier, which only has effect on the first step
* uphill_damp - damp parameter scaled by the angle between the velocity and the force


"""@ref
LAMMPS python routine wrappers
"""

import numpy as np
import sys

def thermo_type(lmp, thermo_ensemble, temperature, pressure, timestep):
    """
    Sets necessary fixes to get the correct thermodynamic ensemble.

    Parameters
    ----------
    lmp            : numpy_wrapper
                     LAMMPS instance.
    thermo_ensemble: str
                     Thermodynamic ensmble "NVE", "NVT", or "NPT".
    temperature    : float
                     Target temperature in Kelvin.
    pressure       : float
                     Target pressure in Bar.
    timestep       : float
                     MD timestep in ps.
    """
    
    if thermo_ensemble == "NVE":
        block = f"""
        fix 1 mobile nve
        fix 2 mobile temp/rescale 1 {temperature} {temperature} 1.0 0.5
        fix_modify 2 temp temp1
        thermo_modify temp temp1
        """ 
    elif thermo_ensemble == "NVT":
        block = f"""
        fix 1 mobile nvt temp {temperature} {temperature} {100.0*timestep}
        fix_modify 1 temp temp1
        thermo_modify temp temp1
        """ 
    elif thermo_ensemble == "NPT":
        block = f"""
        # cannot apply barostat to a non-periodic dimension
        fix 1 mobile npt temp {temperature} {temperature} {100.0*timestep} &
                                x {pressure} {pressure} {1000.0*timestep} &
                                y {pressure} {pressure} {1000.0*timestep}
        fix_modify 1 temp temp1
        thermo_modify temp temp1
        """ 
    else:
        print("Error: thermo_ensemble should be set to 'NVE', 'NVT', or 'NPT'")
        sys.exit()

    lmp.commands_string(block)

# Force calculation

def calculate_force(lmp,sigma,eV_to_J):
    """
    Calculates shear force from the provided shear stress.

    Parameters
    ----------
    lmp     : numpy_wrapper
              LAMMPS instance
    sigma   : float
              Target shear stress in Bar.
    eV_to_J : float
              Conversion factor from eV to Joule.
    """

    block = """
    variable nupper equal count(upper)
    variable nlower equal count(lower)
    variable length_x equal "lx"
    variable length_y equal "ly"
    """
    lmp.commands_string(block)

    nupper = lmp.extract_variable("nupper")
    nlower = lmp.extract_variable("nlower")
    lx = lmp.extract_variable("length_x")
    ly = lmp.extract_variable("length_y")

    app_force_upper = (lx*ly)/nupper*sigma/eV_to_J
    app_force_lower = (lx*ly)/nlower*sigma/eV_to_J

    return app_force_lower, app_force_upper, nlower, nupper, lx, ly

def set_boundary_conditions(lmp,loading_type,app_force_lower,app_force_upper,force):
    """
    Applies the appropriate fixes to the simulation cell boundaries depending on
    loading option selected. The target force can be ramped or a constant force can
    be applied.

    Parameters
    ----------
    lmp             : numpy_wrapper
                      LAMMPS instance
    loading_type    : str
                      If set to "sym" shear force is applied to top and bottom layers of
                      the simulation cell. If set to "asymm" shear force will only be
                      applied to top layer of cell.
    app_force_lower : float
                      Shear force to apply to bottom of simulation cell in eV/Angstrom
    app_force_upper : float
                      Shear force to apply to top of simulation cell in eV/Angstrom
    force           : str
                      If set to "ramp" the applied force is ramped linearlyover 20 ps.
                      Otherwise applies a constant force
    """
    if force == "ramp":
        lmp.command("variable t equal 'time'")
        lmp.command("variable theta equal '20.0'")
        lmp.command(f"variable forceL equal '-({app_force_lower}*v_t)/v_theta'")
        lmp.command(f"variable forceU equal '({app_force_upper}*v_t)/v_theta'")

        force_lower = '${forceL}'
        force_upper = '${forceU}'
    else:
        force_lower = -app_force_lower
        force_upper = app_force_upper
    
    if loading_type == "asymm":
        block = f"""
        fix 3 upper setforce NULL 0.0 0.0
        fix 4 lower setforce 0.0  0.0 0.0
        
        fix 5 upper aveforce {force_upper} 0.0 0.0 
        
        fix 6 upper rigid group 1 upper
        """
        lmp.commands_string(block)

    elif loading_type == "sym":
        block = f"""
        fix 3 upper setforce NULL 0.0 0.0
        fix 4 lower setforce NULL 0.0 0.0
        
        fix 5 upper aveforce {force_upper} 0.0 0.0 
        fix 6 lower aveforce {force_lower} 0.0 0.0 

        fix 7 boundaries rigid group 2 upper lower
        """
        lmp.commands_string(block)
    else:
        print("Error: loading_type should be set to 'sym' or 'asymm'")
        sys.exit()

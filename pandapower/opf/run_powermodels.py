import os

# from pandapower import pp_dir
from pandapower.converter.powermodels.to_pm import convert_to_pm_structure, dump_pm_json
from pandapower.converter.powermodels.from_pm import read_pm_results_to_net

try:
    import pplog as logging
except ImportError:
    import logging


def _runpm(net, delete_buffer_file=True, pm_file_path = None):  # pragma: no cover
    """
    Converts the pandapower net to a pm json file, saves it to disk, runs a PowerModels.jl julia function and reads
    the results back to the pandapower net

    INPUT
    ----------
    **net** - pandapower net

    OPTIONAL
    ----------
    **delete_buffer_file** (bool, True) - deletes the pm buffer json file if True.


    """
    # convert pandapower to power models file -> this is done in python
    net, pm, ppc, ppci = convert_to_pm_structure(net)
    # call optinal callback function
    if net._options["pp_to_pm_callback"] is not None:
        net._options["pp_to_pm_callback"](net, ppci, pm)
    # writes pm json to disk, which is loaded afterwards in julia
    buffer_file = dump_pm_json(pm, pm_file_path)
    # run power models optimization in julia
    result_pm = _call_powermodels(buffer_file, net._options["julia_file"])
    # read results and write back to net
    read_pm_results_to_net(net, ppc, ppci, result_pm)
    if pm_file_path is None and delete_buffer_file:
        # delete buffer file after calculation
        os.remove(buffer_file)



def _call_powermodels(buffer_file, julia_file):  # pragma: no cover
    try:
        import julia
        from julia import Main
        from julia import Pkg
        from julia import Base
    except ImportError:
        raise ImportError("Please install pyjulia to run pandapower with PowerModels.jl")
        
    try:
        j = julia.Julia()
    except:
        raise UserWarning(
            "Could not connect to julia, please check that Julia is installed and pyjulia is correctly configured")
        
    # Pkg.rm("PandaModels")
    # Pkg.resolve()
    # import two julia scripts and runs powermodels julia_file

    if str(type(Base.find_package("PandaModels"))) == "<class 'NoneType'>":
        print("PandaModels is not exist")
        Pkg.Registry.update()
        Pkg.add(url = "https://github.com/e2nIEE/PandaModels.jl") 
        # Pkg.build()
        Pkg.resolve()
        Pkg.develop("PandaModels")
        Pkg.build()
        Pkg.resolve()
        print("add PandaModels")
      

    print(Base.find_package("PandaModels"))    
    # Pkg_path = Base.find_package("PandaModels").split(".jl")[0]

    # try:
    #     Pkg_path = Base.find_package("PandaModels").split(".jl")[0]
    #     print(Pkg_path)
    # except:
    #     Pkg.add(path = "C:/Users/x230/.julia/dev/PandaModels.jl")
    #     Pkg.build()
    #     Pkg.resolve()
        
    # Pkg.activate(Pkg_path)
    Pkg.activate("PandaModels")
    Pkg.instantiate()
    Pkg.resolve()
    print("activate PandaModels")
    
    try:    
        Main.using("PandaModels")
        print("using PandaModels")
    except ImportError:
        raise ImportError("cannot use PandaModels")    
    # if not os.path.isfile(julia_file):
    #     raise UserWarning("File %s could not be imported" % julia_file)
    Main.buffer_file = buffer_file
    # exec_fun = julia_file.split("/")[-1].split(".")[0]
    # print(">>>>>> julia_file: " +  julia_file)  
    result_pm = Main.eval(julia_file+"(buffer_file)")
    return result_pm

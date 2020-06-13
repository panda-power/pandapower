
module PP2PM
export load_pm_from_json, get_model, get_solver

import JSON
using PowerModels

import Cbc
import Ipopt
import Juniper
import JuMP


try
    import Gurobi
catch e
    if isa(e, LoadError)
        println("Cannot import Gurobi. That's fine if you do not plan to use it")
    end
end

try
    import KNITRO
catch e
    if isa(e, LoadError)
        println("Cannot import KNITRO. That's fine if you do not plan to use it")
    end
end

try
    import SCIP
catch e
    if isa(e, LoadError)
        println("Cannot import SCIP. That's fine if you do not plan to use it")
    end
end

function get_model(model_type)
    """
    gets the model function
    model_type (str) - examples: "ACPPowerModel", "DCPPowerModel", "SOCWRPowerModel"...
    see: https://lanl-ansi.github.io/PowerModels.jl/stable/formulation-details/
    """

    s = Symbol(model_type)
    return getfield(Main, s)
end

function get_solver(optimizer::String, nl::String="ipopt", mip::String="cbc",
    log_level::Int=0, time_limit::Float64=Inf, nl_time_limit::Float64=Inf, 
    mip_time_limit::Float64=Inf)
    
    if optimizer == "gurobi"
            solver = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => time_limit, "OutputFlag" => log_level)
    end

    if optimizer == "ipopt"
                solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => log_level, "max_cpu_time" => time_limit)
    end

    if optimizer == "juniper" && nl == "ipopt" && mip == "cbc"
        mip_solver = JuMP.optimizer_with_attributes(Cbc.Optimizer, "logLevel" => log_level, "seconds" => mip_time_limit)
        nl_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer, "print_level" => log_level, "max_cpu_time" => nl_time_limit)
        solver = JuMP.optimizer_with_attributes(Juniper.Optimizer,
                     "nl_solver" => nl_solver,
                     "mip_solver" => mip_solver,
                     "log_levels" => [],
                     "time_limit" => time_limit)
    end

    if optimizer == "juniper" && nl == "gurobi" && mip == "cbc"
        mip_solver = JuMP.optimizer_with_attributes(Cbc.Optimizer, "logLevel" => log_level, "seconds" => mip_time_limit)
        nl_solver = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => nl_time_limit)
        solver = JuMP.optimizer_with_attributes(Juniper.Optimizer,
                     "nl_solver" => nl_solver,
                     "mip_solver" => mip_solver,
                     "log_levels" => [],
                     "time_limit" => time_limit)
    end

    if optimizer == "juniper" && nl == "gurobi" && mip == "gurobi"
        mip_solver = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => mip_time_limit)
        nl_solver = JuMP.optimizer_with_attributes(Gurobi.Optimizer, "TimeLimit" => nl_time_limit)
        solver = JuMP.optimizer_with_attributes(Juniper.Optimizer,
                     "nl_solver" => nl_solver,
                     "mip_solver" => mip_solver,
                     "log_levels" => [],
                     "time_limit" => time_limit)
    end

    if optimizer == "knitro"
        solver = JuMP.optimizer_with_attributes(KNITRO.Optimizer)
    end

    if optimizer == "cbc"
        solver = JuMP.optimizer_with_attributes(Cbc.Optimizer, "seconds" => time_limit)
    end

    if optimizer == "scip"
        solver = JuMP.optimizer_with_attributes(SCIP.Optimizer)
    end

    return solver

end

function load_pm_from_json(json_path)
    pm = Dict()
    open(json_path, "r") do f
        pm = JSON.parse(f)  # parse and transform data
    end

    for (idx, gen) in pm["gen"]
        if gen["model"] == 1
            pm["gen"][idx]["cost"] = convert(Array{Float64,1}, gen["cost"])
        end
    end
    if pm["correct_pm_network_data"]
        correct_network_data!(pm)
    end
    return pm
end

end

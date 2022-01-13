import os

from exptools.collections import AttrDict
from exptools.launching.cluster import ClusterHandlerBase

# Readable-to-less-readable abbreviations.
N_GPU = "gpu"
CONTEXTS_PER_GPU = "cxg"  # CUDA contexts.
GPU_PER_RUN = "gpr"
N_CPU_CORE = "cpu"
HYPERTHREAD_OFFSET = "hto"  # Can specify if different from n_cpu_core.
N_SOCKET = "skt"
RUN_SLOT = "slt"
CPU_PER_WORKER = "cpw"
CPU_PER_RUN = "cpr"  # For cpu-only.
CPU_RESERVED = "res"  # Reserve CPU cores per master, not allowed by workers.
# For async sampling / optimizing.
ASYNC_SAMPLE = "ass"
SAMPLE_GPU_PER_RUN = "sgr"
OPTIM_SAMPLE_SHARE_GPU = "oss"
# For alternating sampler.
ALTERNATING = "alt"
SET_AFFINITY = "saf"

ABBREVS = [N_GPU, CONTEXTS_PER_GPU, GPU_PER_RUN, N_CPU_CORE,
    HYPERTHREAD_OFFSET, N_SOCKET, CPU_PER_RUN, CPU_PER_WORKER, CPU_RESERVED,
    ASYNC_SAMPLE, SAMPLE_GPU_PER_RUN, OPTIM_SAMPLE_SHARE_GPU,
    ALTERNATING, SET_AFFINITY]


# API

def quick_affinity_code(n_parallel=None, use_gpu=True, contexts_per_gpu= 1):
    if not (use_gpu or n_parallel):
        raise ValueError("Either use_gpu must be True or n_parallel > 0 must be given.")
    import psutil
    n_cpu_core = psutil.cpu_count(logical=False)
    if use_gpu:
        try:
            import torch
            n_gpu = torch.cuda.device_count()
        except ImportError as e:
            n_gpu = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
    else:
        n_gpu = 0
    if n_gpu > 0:
        if n_parallel is not None:
            n_gpu = min(n_parallel, n_gpu)
        n_runs = n_gpu * contexts_per_gpu
        n_cpu_core = (n_cpu_core // n_runs) * n_runs  # Same for all.
        return encode_affinity(n_cpu_core=n_cpu_core, n_gpu=n_gpu,
            contexts_per_gpu= contexts_per_gpu)
    else:
        if not n_parallel:
            raise ValueError(
                "n_parallel > 0 must be given if use_gpu=False or no GPUs are present."
            )
        n_parallel = min(n_parallel, n_cpu_core)
        n_cpu_core = (n_cpu_core // n_parallel) * n_parallel  # Same for all.
        cpu_per_run = n_cpu_core // n_parallel
        return encode_affinity(n_cpu_core=n_cpu_core, n_gpu=0,
            cpu_per_run=cpu_per_run)

def full_resource_affinity(set_affinity= True):
    """ In order to cooperate with cluster manager, assuming all resources it sees are assigned to this
    run of experiment.
    """
    import psutil
    n_cpu_core = psutil.cpu_count(logical=False)
    try:
        import torch
        n_gpu = torch.cuda.device_count()
    except ImportError as e:
        n_gpu = len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) if "CUDA_VISIBLE_DEVICES" in os.environ else 0
    return affinity_from_code(encode_affinity(
            n_cpu_core= n_cpu_core,
            n_gpu= n_gpu,
            cpu_per_run= n_cpu_core,
            gpu_per_run= max(1, n_gpu),
            run_slot= 0,
            set_affinity= set_affinity,
        ))

def encode_affinity(
        n_cpu_core=None,  # Total number to use on machine (not virtual).
        n_gpu=0,  # Total number to use on machine.
        cpu_reserved=0,  # Number CPU to reserve per GPU.
        contexts_per_gpu=1,  # e.g. 2 will put two experiments per GPU.
        gpu_per_run=1,  # For multi-GPU optimizaion.
        cpu_per_run=1,  # Specify if not using GPU.
        cpu_per_worker=1,  # Use 1 unless environment is multi-threaded.
        async_sample=False,  # True if asynchronous sampling / optimization.
        sample_gpu_per_run=0,  # For asynchronous sampling.
        optim_sample_share_gpu=False,  # Async sampling, overrides sample_gpu.
        hyperthread_offset=None,  # Leave None for auto-detect.
        n_socket=None,  # Leave None for auto-detect.
        run_slot=None,  # Leave None in `run` script, but specified in `train` script.
        alternating=False,  # True for altenating sampler.
        set_affinity=True,  # Everything same except psutil.Process().cpu_affinity(cpus)
        ):
    """Use in run script to specify computer configuration."""
    if n_cpu_core is None: # If not specified, all cores that the launcher can access will be counted.
        import psutil
        n_cpu_core = psutil.cpu_count(logical=False)
    affinity_code = f"{n_cpu_core}{N_CPU_CORE}_{n_gpu}{N_GPU}"
    if hyperthread_offset is None:
        hyperthread_offset = get_hyperthread_offset()
    if n_socket is None:
        n_socket = get_n_socket()
    if contexts_per_gpu > 1:
        affinity_code += f"_{contexts_per_gpu}{CONTEXTS_PER_GPU}"
    if gpu_per_run > 1:
        affinity_code += f"_{gpu_per_run}{GPU_PER_RUN}"
    if n_gpu == 0:
        affinity_code += f"_{cpu_per_run}{CPU_PER_RUN}"
    if cpu_per_worker > 1:
        affinity_code += f"_{cpu_per_worker}{CPU_PER_WORKER}"
    if hyperthread_offset != n_cpu_core:
        affinity_code += f"_{hyperthread_offset}{HYPERTHREAD_OFFSET}"
    if n_socket > 1:
        affinity_code += f"_{n_socket}{N_SOCKET}"
    if cpu_reserved > 0:
        affinity_code += f"_{cpu_reserved}{CPU_RESERVED}"
    if async_sample:
        affinity_code += f"_1{ASYNC_SAMPLE}"
    if sample_gpu_per_run > 0:
        affinity_code += f"_{sample_gpu_per_run}{SAMPLE_GPU_PER_RUN}"
    if optim_sample_share_gpu:
        affinity_code += f"_1{OPTIM_SAMPLE_SHARE_GPU}"
    if alternating:
        affinity_code += f"_1{ALTERNATING}"
    if not set_affinity:
        affinity_code += f"_0{SET_AFFINITY}"
    if run_slot is not None:
        assert run_slot <= (n_gpu * contexts_per_gpu) // gpu_per_run
        affinity_code = f"{run_slot}{RUN_SLOT}_" + affinity_code
    return affinity_code


def prepend_run_slot(run_slot, affinity_code):
    """Use in launch manager when assigning run slot."""
    return f"{run_slot}{RUN_SLOT}_" + affinity_code


def affinity_from_code(run_slot_affinity_code):
    """Use in individual experiment script; pass output to Runner."""
    if run_slot_affinity_code == ClusterHandlerBase.affinity_code:
        # to support cluster manager acquire all resources this job can access
        return full_resource_affinity(set_affinity= False)
    run_slot, aff_code = remove_run_slot(run_slot_affinity_code)
    aff_params = decode_affinity(aff_code)
    if aff_params.get(N_GPU, 0) > 0:
        if aff_params.pop(ASYNC_SAMPLE, 0) > 0:
            return build_async_affinity(run_slot, **aff_params)
        elif aff_params.get(GPU_PER_RUN, 1) > 1:
            return build_multigpu_affinity(run_slot, **aff_params)
        return build_gpu_affinity(run_slot, **aff_params)
    return build_cpu_affinity(run_slot, **aff_params)


def make_affinity(run_slot=0, **kwargs):
    """ Input same kwargs as encode_affinity, returns the AttrDict form.
    
        Affinity: a dictionary-like configuration tells you which devices 
        are used to run this experiment.
    """
    return affinity_from_code(encode_affinity(run_slot=run_slot, **kwargs))


# Helpers

def get_n_socket():
    import subprocess
    return max(1, int(subprocess.check_output(
        'cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l',
        shell=True)))


def get_hyperthread_offset():
    import psutil  # (If returns 0, will not try to use hyperthreads.)
    # UNRELIABLE:
    # hto = psutil.cpu_count() - psutil.cpu_count(logical=False)
    vcpu = psutil.cpu_count()
    if vcpu != psutil.cpu_count(logical=False) and vcpu % 2 == 0:
        # Best guess?
        return vcpu // 2
    return 0


def get_n_run_slots(affinity_code):
    """ Return the number of experiment run in parallel
    """
    aff = decode_affinity(affinity_code)
    if aff.get("ass", 0) > 0:  # Asynchronous sample mode.
        total_gpu = aff.get("gpr", 1) + aff.get("sgr", 0) * (1 - aff.get("oss", 0))
        n_run_slots = aff["gpu"] // total_gpu  # NOTE: no cxg yet.
    elif aff.get("gpu", 0) > 0:
        n_run_slots = (aff["gpu"] * aff.get("cxg", 1)) // aff.get("gpr", 1)
    else:
        n_run_slots = aff["cpu"] // aff["cpr"]
    return n_run_slots


def remove_run_slot(run_slot_affinity_code):
    run_slot_str, aff_code = run_slot_affinity_code.split("_", 1)
    assert run_slot_str[-3:] == RUN_SLOT
    run_slot = int(run_slot_str[:-3])
    return run_slot, aff_code


def decode_affinity(affinity_code):
    codes = affinity_code.split("_")
    aff_kwargs = dict()
    for code in codes:
        abrv = code[-3:]
        if abrv not in ABBREVS:
            raise ValueError(f"Unrecognized affinity code abbreviation: {abrv}")
        value = int(code[:-3])
        aff_kwargs[abrv] = value
    return aff_kwargs


def build_cpu_affinity(slt, cpu, cpr, cpw=1, hto=None, res=0, skt=1, gpu=0,
        alt=0, saf=1):
    assert gpu == 0
    assert cpu % cpr == 0
    hto = cpu if hto is None else hto  # Default is None, 0 is OFF.
    assert (hto - cpu) % skt == 0
    n_run_slots = cpu // cpr
    assert slt <= n_run_slots
    cpu_per_skt = max(cpu, hto) // skt
    if n_run_slots >= skt:
        slt_per_skt = n_run_slots // skt
        my_skt = slt // slt_per_skt
        slt_in_skt = slt % slt_per_skt
        min_core = my_skt * cpu_per_skt + slt_in_skt * cpr
        cores = tuple(range(min_core, min_core + cpr))
    else:  # One run multiple sockets.
        skt_per_slt = skt // n_run_slots
        cores = list()
        low_skt = slt * skt_per_slt
        for s in range(skt_per_slt):
            min_core = (low_skt + s) * cpu_per_skt
            high_core = min_core + cpr // skt_per_slt
            cores.extend(list(range(min_core, high_core)))
        cores = tuple(cores)
    worker_cores = cores[res:]
    assert len(worker_cores) % cpw == 0
    master_cpus = get_master_cpus(cores, hto)
    workers_cpus = get_workers_cpus(worker_cores, cpw, hto, alt)
    affinity = AttrDict(
        all_cpus=master_cpus,
        master_cpus=master_cpus,
        workers_cpus=workers_cpus,
        # NOTE: see git log for the missing *_torch_threads
        alternating=bool(alt),  # Just to pass through a check.
        set_affinity=bool(saf),
    )
    return affinity


def build_gpu_affinity(slt, gpu, cpu, cxg=1, cpw=1, hto=None, res=0, skt=1,
        alt=0, saf=1):
    """Divides CPUs evenly among GPUs."""
    n_ctx = gpu * cxg
    assert slt <= n_ctx
    assert cpu % n_ctx == 0
    cpr = cpu // n_ctx
    if cxg > 1:
        slt = (slt % gpu) * cxg + slt // gpu  # Spread over GPUs first.
    affinity = build_cpu_affinity(slt, cpu, cpr, cpw, hto, res, skt, alt, saf)
    affinity["cuda_idx"] = slt // cxg # the GPU index among all GPUs visible to this process.
    return affinity


def build_multigpu_affinity(run_slot, gpu, cpu, gpr=1, cpw=1, hto=None, res=0,
        skt=1, alt=0, saf=1):
    return [build_gpu_affinity(slt, gpu, cpu, cxg=1, cpw=cpw, hto=hto, res=res,
        skt=skt, alt=alt, saf=saf) for slt in range(run_slot * gpr, (run_slot + 1) * gpr)]


def build_async_affinity(run_slot, gpu, cpu, gpr=1, sgr=0, oss=0, cpw=1,
        hto=None, res=1, skt=1, alt=0, saf=1):
    oss = bool(oss)
    sgr = gpr if oss else sgr
    total_gpr = (gpr + sgr * (not oss))
    n_run_slots = gpu // total_gpr
    assert run_slot < n_run_slots
    cpr = cpu // n_run_slots
    smp_cpr = cpr - res * gpr
    gpu_per_skt = gpu // skt
    hto = cpu if hto is None else hto  # Default is None, 0 is OFF.
    cpu_per_skt = max(cpu, hto) // skt
    opt_affinities = list()
    smp_affinities = list()
    all_cpus = tuple()
    if total_gpr <= gpu_per_skt:
        run_per_skt = n_run_slots // skt
        assert n_run_slots % skt == 0  # Relax later?
        skt_per_run = 1
        run_in_skt = run_slot % run_per_skt
        my_skt = run_slot // run_per_skt
        low_opt_gpu = my_skt * gpu_per_skt + run_in_skt * total_gpr
        high_opt_gpu = low_opt_gpu + gpr
        my_opt_gpus = list(range(low_opt_gpu, high_opt_gpu))
        my_smp_gpus = (my_opt_gpus if oss else
            list(range(high_opt_gpu, high_opt_gpu + sgr)))
    else:  # One run takes more than one socket: spread opt gpus across sockets.
        skt_per_run = skt // n_run_slots
        low_skt = run_slot * skt_per_run
        assert gpr % skt_per_run == 0, "Maybe try n_socket=1."
        assert sgr % skt_per_run == 0, "Maybe try n_socket=1."
        my_opt_gpus = list()
        my_smp_gpus = list()
        run_in_skt = run_per_skt = 0
        for s in range(skt_per_run):
            low_opt_gpu = (low_skt + s) * gpu_per_skt
            high_opt_gpu = low_opt_gpu + gpr // skt_per_run
            my_opt_gpus.extend(list(range(low_opt_gpu, high_opt_gpu)))
            if oss:
                my_smp_gpus = my_opt_gpus
            else:
                high_smp_gpu = high_opt_gpu + sgr // skt_per_run
                my_smp_gpus.extend(list(range(high_opt_gpu, high_smp_gpu)))
    for i, opt_gpu in enumerate(my_opt_gpus):
        gpu_in_skt = opt_gpu % gpu_per_skt
        gpu_skt = opt_gpu // gpu_per_skt
        gpu_res = i if run_per_skt >= 1 else gpu_in_skt
        low_opt_core = (gpu_skt * cpu_per_skt + run_in_skt * cpr +
            gpu_res * res)
        high_opt_core = low_opt_core + res
        opt_cores = tuple(range(low_opt_core, high_opt_core))
        opt_cpus = get_master_cpus(opt_cores, hto)
        opt_affinity = dict(
            cpus=opt_cpus,
            cuda_idx=opt_gpu,         
            # NOTE: see git log for the missing *_torch_threads
            set_affinity=bool(saf),
        )
        opt_affinities.append(opt_affinity)
        all_cpus += opt_cpus
    wrkr_per_smp = smp_cpr // cpw
    smp_cpr = wrkr_per_smp * cpw
    smp_cpg = smp_cpr // max(1, sgr)
    for i, smp_gpu in enumerate(my_smp_gpus):
        gpu_skt = smp_gpu // gpu_per_skt
        gpu_in_skt = smp_gpu % gpu_per_skt
        smp_cpu_off = (i if run_per_skt >= 1 else
            gpu_in_skt - (gpr // skt_per_run))
        low_smp_core = (gpu_skt * cpu_per_skt + run_in_skt * cpr +
            (gpr // skt_per_run) * res + smp_cpu_off * smp_cpg)
        high_smp_core = low_smp_core + smp_cpg
        master_cores = tuple(range(low_smp_core, high_smp_core))
        master_cpus = get_master_cpus(master_cores, hto)
        workers_cpus = get_workers_cpus(master_cores, cpw, hto, alt)
        smp_affinity = AttrDict(
            all_cpus=master_cpus,
            master_cpus=master_cpus,
            workers_cpus=workers_cpus,
            # NOTE: see git log for the missing *_torch_threads
            cuda_idx=smp_gpu,
            alternating=bool(alt),  # Just to pass through a check.
            set_affinity=bool(saf),
        )
        smp_affinities.append(smp_affinity)
        all_cpus += master_cpus
    if not smp_affinities:  # sgr==0; CPU sampler.
        if total_gpr <= gpu_per_skt:
            low_smp_core = (my_skt * cpu_per_skt + run_in_skt * cpr +
                gpr * res)
            master_cores = tuple(range(low_smp_core, low_smp_core + smp_cpr))
        else:
            master_cores = tuple()
            for s in range(skt_per_run):
                low_smp_core = ((low_skt + s) * cpu_per_skt +
                    (gpr // gpu_per_skt) * res)
                master_cores += tuple(range(low_smp_core, low_smp_core +
                    smp_cpr // skt_per_run))
        master_cpus = get_master_cpus(master_cores, hto)
        workers_cpus = get_workers_cpus(master_cores, cpw, hto, alt)
        smp_affinities = AttrDict(
            all_cpus=master_cpus,
            master_cpus=master_cpus,
            workers_cpus=workers_cpus,
            # NOTE: see git log for the missing *_torch_threads
            cuda_idx=None,
            alternating=bool(alt),  # Just to pass through a check.
            set_affinity=bool(saf),
        )
        all_cpus += master_cpus
    affinity = AttrDict(
        all_cpus=all_cpus,  # For exp launcher to use taskset.
        optimizer=opt_affinities,
        sampler=smp_affinities,
        set_affinity=bool(saf),
    )

    return affinity

def get_master_cpus(cores, hto):
    hyperthreads = tuple(c + hto for c in cores) if hto > 0 else ()
    return tuple(cores) + hyperthreads


def get_workers_cpus(cores, cpw, hto, alt):
    cores = cores[:(len(cores) // cpw) * cpw]  # No worker less than cpw.
    cpus = tuple(cores[i:i + cpw]
        for i in range(0, len(cores), cpw))
    if hto > 0:
        hyperthreads = tuple(c + hto for c in cores)
        hyperthreads = tuple(hyperthreads[i:i + cpw]
            for i in range(0, len(cores), cpw))
        if alt:
            cpus += hyperthreads
        else:
            cpus = tuple(c + h for c, h in zip(cpus, hyperthreads))
    elif alt:
        cpus += cpus
    return cpus

def set_gpu_from_visibles(cuda_idxs, n_gpu_max= 16):
    """ NOTE: we assume a machine has at most 16 gpu installed (system wise)
    Given the cuda idx, we set the environment variable from original
    cuda variable if existed. In order to meet the multi-tasks GPU usage.
    """
    if not isinstance(cuda_idxs, list):
        cuda_idxs = [cuda_idxs]

    try:
        env_devices = os.environ['CUDA_VISIBLE_DEVICES']
        assert env_devices != ""
    except:
        env_devices = ",".join(str(range(n_gpu_max)))
    env_idxs = [s for s in env_devices.split(",")]

    set_idxs = [] # a list of str
    for cuda_idx in cuda_idxs:
        set_idxs.append(env_idxs[cuda_idx])

    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(set_idxs)

def combine_affinity(affinities):
        """ Considering the affinity could be a list of affinity, conbine all resources at the
        the disposal of one experiment
        """
        affinity = {f: list() for f in affinities[0].keys()}
        for k in affinity.keys():
            if "cpus" in k:
                for aff in affinities:
                    affinity[k].extend(aff[k])
            elif "torch_threads" in k:
                num = 0
                for aff in affinities:
                    num += aff[k]
                affinity[k] = num
            elif "cuda_idx" == k:
                for aff in affinities:
                    affinity[k].append(aff[k])
            else:
                # should be "alternating" and "set_affinity" keys
                affinity[k] = False
                for aff in affinities:
                    affinity[k] |= aff[k]
        return affinity

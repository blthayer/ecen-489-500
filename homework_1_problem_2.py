########################################################################
# IMPORTS
########################################################################
from constraint import Problem

########################################################################
# CLASSES
########################################################################


class Case:
    """Computer case class."""
    def __init__(self, case_id, motherboard_factors, pci_max_length):
        self.id = case_id
        self.motherboard_factors = motherboard_factors
        self.pci_max_length = pci_max_length


class Motherboard:
    """Motherboard class."""
    def __init__(self, motherboad_id, factor, chipset, dimm_sockets,
                 nvme_slots):
        self.id = motherboad_id
        self.factor = factor
        self.chipset = chipset
        self.dimm_sockets = dimm_sockets
        self.nvme_slots = nvme_slots

        if self.chipset == 'Z370':
            self.cpu_overclock = True
        else:
            self.cpu_overclock = False


class Memory:
    """Memory class."""
    def __init__(self, memory_id, num):
        self.id = memory_id
        self.num = num


class GPU:
    """GPU class."""
    def __init__(self, gpu_id, pci_length):
        self.id = gpu_id
        self.pci_length = pci_length


class SSD:
    """Storage (SSD) class."""
    def __init__(self, ssd_id, nvme):
        """
        :param ssd_id: ID of the SSD.
        :param nvme: 1/0 --> 1 if the SSD is nVME, 0 if the SSD is SATA
        """
        self.id = ssd_id
        self.nvme = nvme


class CPU:
    """Processor (CPU) class."""
    def __init__(self, cpu_id, overclock):
        self.id = cpu_id
        self.overclock = overclock

########################################################################
# CONSTANTS FOR PROBLEM 2
########################################################################


MINI = 'miniITX'
MICRO = 'microATX'

CASES = [
    Case(case_id='SS', motherboard_factors=[MINI], pci_max_length=9),
    Case(case_id='V21', motherboard_factors=[MINI, MICRO],
         pci_max_length=11.2)
]

MOTHERBOARDS = [
    Motherboard(motherboad_id='H310ITX', factor=MINI, chipset='H310',
                dimm_sockets=2, nvme_slots=0),
    Motherboard(motherboad_id='B360ATX', factor=MICRO, chipset='B360',
                dimm_sockets=4, nvme_slots=2),
    Motherboard(motherboad_id='Z370ITX', factor=MINI, chipset='Z370',
                dimm_sockets=2, nvme_slots=2)
]

MEMORY = [
    Memory(memory_id='2x8', num=2),
    Memory(memory_id='4x4', num=4)
]

GPUS = [
    GPU(gpu_id='GTXlong', pci_length=9.53),
    GPU(gpu_id='GTXshort', pci_length=8.8)
]

SSDS = [
    SSD(ssd_id='nVME', nvme=1),
    SSD(ssd_id='SSD', nvme=0)
]

CPUS = [
    CPU(cpu_id='i7', overclock=True),
    CPU(cpu_id='i5', overclock=False)
]

########################################################################
# FUNCTIONS
########################################################################


def motherboard_case(motherboard, case):
    """Determine if a Motherboard and Case are compatible (True) or not
    (False).
    """
    if motherboard.factor in case.motherboard_factors:
        return True
    else:
        return False


def motherboard_memory(motherboard, memory):
    """Determine if a Motherboard and Memory are compatible (True) or
    not (False).
    """
    if motherboard.dimm_sockets < memory.num:
        return False
    else:
        return True


def motherboard_ssd(motherboard, ssd):
    """Determine if a Motherboard and SSD are compatible (True) or not
    (False).
    """
    if ssd.nvme > motherboard.nvme_slots:
        return False
    else:
        return True


def motherboard_cpu(motherboard, cpu):
    """Determine if a Motherboard and CPU are compatible (True) or not
    (False).
    """
    if cpu.overclock:
        if motherboard.cpu_overclock:
            return True
        else:
            return False
    else:
        return True


def case_gpu(case, gpu):
    """Determine if a Case and GPU are compatible (True) or not (False).
    """
    if gpu.pci_length > case.pci_max_length:
        return False
    else:
        return True


def problem_2():
    """Function for performing the work of problem 2."""
    # Initialize the problem.
    problem = Problem()

    # Add the variables.
    problem.addVariable('case', CASES)
    problem.addVariable('motherboard', MOTHERBOARDS)
    problem.addVariable('memory', MEMORY)
    problem.addVariable('GPU', GPUS)
    problem.addVariable('SSD', SSDS)
    problem.addVariable('CPU', CPUS)

    # Add constraints.
    problem.addConstraint(motherboard_case, ['motherboard', 'case'])
    problem.addConstraint(motherboard_memory, ['motherboard', 'memory'])
    problem.addConstraint(motherboard_ssd, ['motherboard', 'SSD'])
    problem.addConstraint(motherboard_cpu, ['motherboard', 'CPU'])
    problem.addConstraint(case_gpu, ['case', 'GPU'])

    return problem


def generate_pc_id(solution):
    variables = ['case', 'motherboard', 'CPU', 'memory', 'SSD', 'GPU']
    out = ''
    for v in variables:
        out += solution[v].id

    return out

########################################################################
# MAIN
########################################################################


if __name__ == '__main__':
    pdf = 'homework_1.pdf'
    # Problem 2
    print('Problem 2:')
    problem_2()
    print('See {} for part (a)'.format(pdf))
    p2 = problem_2()
    sol2 = p2.getSolution()
    print('(b):')
    print('Valid configuration:')
    print(generate_pc_id(sol2))
    all_sol2 = p2.getSolutions()
    print('There are {} valid solutions.'.format(len(all_sol2)))

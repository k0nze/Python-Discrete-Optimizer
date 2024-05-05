import os
import pickle
import signal
import sys
import logging
from pprint import pprint

import acadl.examples.architectures.vta as vta

from acadl import clear_names, latency_t, AIDG, evaluation_method_t, Instruction
from acadl.examples.instruction_sets.vta_instructions import VTAInstructionParser
from acadl.functional_simulation import FunctionalSimulator, ArchitecturalState


__path_to_keras_traces__ = "/local/luebeck/keras_vta_traces"
__txt_trace_file_paths_with_errors__ = []

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def signal_handler(sig, frame):
    print("\nCtrl+C detected. Performing cleanup before exiting.")

    if len(__txt_trace_file_paths_with_errors__) != 0:
        pprint(__txt_trace_file_paths_with_errors__)

        with open('txt_trace_file_paths_with_errors.txt', 'w') as f:
            for txt_trace_file_path_with_errors in __txt_trace_file_paths_with_errors__:
                f.write(f"{txt_trace_file_path_with_errors}\n")

    sys.exit(0)

# Set the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":

    ag = vta.get_ag()

    # collect all txt trace file paths
    txt_trace_file_names  = os.listdir(f"{__path_to_keras_traces__}/txt")
    txt_trace_file_paths = []

    for txt_trace_file_name in txt_trace_file_names:
        txt_trace_file_path = f"{__path_to_keras_traces__}/txt/{txt_trace_file_name}"
        txt_trace_file_paths.append(txt_trace_file_path)


    for txt_trace_file_path in txt_trace_file_paths[:1]:
        logging.info(f"Start converting '{txt_trace_file_path}'")

        instructions = []

        # parse instructions
        try:
            with open(txt_trace_file_path) as f:
                instructions_str = f.read()
            parser = VTAInstructionParser(instructions_str)
            uops = parser.parse_uops()
            instructions = parser.parse()       
        except Exception as e:
            __txt_trace_file_paths_with_errors__.append(txt_trace_file_path)
            logging.error(f"Parsing '{txt_trace_file_path}': " + str(e))

        logging.info(f"Done parsing '{txt_trace_file_path}'")

        if len(instructions) == 0:
            __txt_trace_file_paths_with_errors__.append(txt_trace_file_path)
            logging.error(f"No instructions parsed in '{txt_trace_file_path}'")
            continue

        traced_instruction_list = []

        # simulate instructions
        try:
            fs = FunctionalSimulator(
                ag,
                instructions=instructions,
                memory_initialization={},
            )

            fs.simulate()
        
            traced_instruction_list = fs.instruction_trace.instructions
        except Exception as e:
            __txt_trace_file_paths_with_errors__.append(txt_trace_file_path)
            logging.error(f"Simulating '{txt_trace_file_path}': " + str(e))

        logging.info(f"Done simulating '{txt_trace_file_path}'")

        if len(traced_instruction_list) == 0:
            logging.error(f"No instructions traced in '{txt_trace_file_path}'")
            continue

        # pickle
        try:
            pickle_trace_file_path = txt_trace_file_path.replace("txt", "pickle")

            with open(pickle_trace_file_path, "wb") as f:
                pickle.dump(traced_instruction_list, f)
        except Exception as e:
            __txt_trace_file_paths_with_errors__.append(txt_trace_file_path)
            logging.error(f"Pickling '{txt_trace_file_path}': " + str(e))


    # persist trace file paths with caused errors during conversion
    signal_handler(None, None) 

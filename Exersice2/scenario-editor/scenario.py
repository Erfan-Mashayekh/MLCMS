
import subprocess
import shlex
import json

class Scenario:
    """
    Run the current scenarios.
    """

    def __init__(self, primary_scenario, edited_scenario, method):
        self.method = method
        self.primary_scenario = primary_scenario
        self.edited_scenario = edited_scenario


    def set_directory(self):
        """
        Set the direcoties of the scenario and the output file.
        """
        primary_address = "../Scenarios/ModelTests/Test" + self.method + "/scenarios/" + self.primary_scenario + ".scenario"
        edited_address = "../Scenarios/ModelTests/Test" + self.method + "/scenarios/" + self.edited_scenario + ".scenario"
        output_dir = "../Scenarios/ModelTests/Test" + self.method + "/output/"
        return primary_address, edited_address, output_dir


    def read_scenario(self, primary):
        """
        Read the scenario.
        primary = True : reads the primary sceanrio file
        primary = False : reads the edited sceanrio file
        """
        if primary:
            scenario_address = self.set_directory()[0]
        else:
            scenario_address = self.set_directory()[1] 

        with open (scenario_address, "r") as f:
            scenario_dict = json.load(f)
        
        return scenario_dict


    def write_scenario(self, scenario_dict):
        edited_address = self.set_directory()[1]
        with open (edited_address, "w") as f:
            json.dump(scenario_dict, f, indent=2)


    def run_scenario(self, pedestrian):
        """
        Run the scenario.
        """
        print("\n-------------------------- Running the Simulation:\n")
        if pedestrian.primary == True:
            scenario_address = self.set_directory()[0]
        else:
            scenario_address = self.set_directory()[1]
        output_dir = self.set_directory()[2]
        run_command = "java -jar ../vadere-console.jar scenario-run --scenario-file " + scenario_address + " --output-dir " + output_dir
        result = subprocess.run(shlex.split(run_command), capture_output=True, text=True)
        print(result.stdout)


    def choices(self):
        """
        Print different avialable options in the code.
        """
        print("\n--------------------------------------------------")
        print("(i) View Pedestrians Info")
        print("(a) Add Pedestrian")
        print("(d) Delete Pedestrian")
        print("(e) Edit Pedestrian (Not implemented yet)")
        print("(r) Run Simulation")
        print("( ) Exit")
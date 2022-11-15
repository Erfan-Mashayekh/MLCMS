
from scenario import Scenario
from pedestrian import Pedestrian


if __name__ == '__main__':

    method = "SIR"                          # Set model
    primary_scenario = "SIR-1" # Read scenario from here
    edited_scenario = "SIR-4"  # Save scenario here

    scenario = Scenario(primary_scenario, edited_scenario, method)
    pedestrian = Pedestrian()

    while True:
        scenario.choices()
        choice = input("\nEnter Key: ")
        if choice == "i":
            pedestrian.print_info(scenario)
        elif choice == "a":
            pedestrian.add(scenario)
        elif choice == "d":
            pedestrian.delete(scenario)
        elif choice == "e":
            pedestrian.edit(scenario)
        elif choice == "r":    
            scenario.run_scenario(pedestrian)
        else:
            print(f"Exited: No option with key: {choice}")
            exit()
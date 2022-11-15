
import json
import numpy as np
import copy

class Pedestrian:
    """
    Checks pedestrians' information.
    """

    def __init__(self):
        self.id = []    # id of the dynamic pedestrian
        self.scenario_dict = None # Scenario file dictionary
        self.dynamicElements = None
        self.primary = True # (True):primary scenario, (False):edited scenario

    def get_info(self, scenario):
        """
        Get the information related to dynamic pedestrian from the scenario file.
        primary = True : primary scenario
        primary = False : edited scenario
        """
        self.id = []
        self.scenario_dict = scenario.read_scenario(self.primary)
        self.dynamicElements = self.scenario_dict["scenario"]["topography"]["dynamicElements"]

        for entry in self.dynamicElements:                
            attributes = entry["attributes"]
            self.id.append(attributes["id"])

    def print_info(self, scenario):
        """
        Print the information of the dynamic pedestrians
        """
        self.get_info(scenario)
        print("\n--------- Getting dyanamic pedestrian information:\n")
        if not self.dynamicElements:
            print(f"dynamicElements is empty.\n")
        else:
            print(f"Dynamic elements attributes are:\n")
            for entry in self.dynamicElements:                
                attributes = entry["attributes"]
                position = entry["position"]
                self.id.append(attributes["id"])

                print(f" Id : ", attributes["id"])
                print(f" Position : ", position["x"],",", position["y"])

    def get_topography_bounds(self):
        """
        Get the topography width and height limits
        #TODO: Consider height in computations
        """
        width = self.scenario_dict["scenario"]["topography"]["attributes"]["bounds"]["width"]
        height = self.scenario_dict["scenario"]["topography"]["attributes"]["bounds"]["height"]
        boundingBoxWidth = 0.5 + self.scenario_dict["scenario"]["topography"]["attributes"]["boundingBoxWidth"]
        assert width == height , f"Error: width and height must be equal."

        return width, boundingBoxWidth

    def add(self, scenario):
        """
        Add a dynamic pedestrian.
        This function reads the 'pedestrian.json' file. You can edit the .json file directly.
        There is no need to set the id of the pedestrian in the json file.
        There is no need to set the targetId in the json file.
        """
        self.get_info(scenario)
        self.primary = False

        print("How to add pedestrians")
        print("(a) Automatically")
        print("(m) Manually")
        switch = input("Enter key: ")
        assert switch == "a" or "m", f"Error: key must be either 'm' or 'a'."

        with open ("pedestrian.json", "r") as f:
            pedestrian_data = json.load(f)
        
        # Generate random positions inside the area in automatic case
        if switch == "a":
            pedestrians_number = int(input("Enter the number of pedestrians to be added: "))
            print("\n--------------------------- Adding the pedestrian automatically:\n")
            width, boundingBoxWidth = self.get_topography_bounds()
            position = (width - 2 * boundingBoxWidth) * np.random.rand(pedestrians_number, 2) + boundingBoxWidth
        elif switch == "m":
            print("\n--------------------------- Adding the pedestrian manually:\n")
            pedestrians_number = 1

        for i in range(pedestrians_number):
            # set an id for the pedestrian
            if not self.id: 
                self.id.append(1)
                pedestrian_data["attributes"]["id"] = 1
            else:
                pedestrian_data["attributes"]["id"] = max(self.id) + 1
                self.id.append(max(self.id) + 1)
            # set random position for pedestrians in case of automatic pedestrian creation
            if switch == "a":
                pedestrian_data["position"]["x"] = position[i][0] 
                pedestrian_data["position"]["y"] = position[i][1]
            self.dynamicElements.append(copy.deepcopy(pedestrian_data)) # add the data from json file        
        
        # find and set the target Ids from the scenario file
        # TODO: So far all pedestrians move to a specified target but can define targetIds automatically
        targetIds = []
        sources = self.scenario_dict["scenario"]["topography"]["sources"]
        for entry in sources:
            targetIds.append(entry["targetIds"])
        if not targetIds:
            pass
        else:
            targetIds = input(f"Choose target id from list {targetIds}. Only type the number: ")
            for entry in self.dynamicElements:
                entry["targetIds"] = [int(targetIds)]

        # change the name of the scenario
        self.scenario_dict["name"] = scenario.edited_scenario

        # write the pedestrian info to scenario file
        self.scenario_dict["scenario"]["topography"]["dynamicElements"] = self.dynamicElements
        scenario.write_scenario(self.scenario_dict)
        if switch == "m":
            pedestrian_id = pedestrian_data["attributes"]["id"]
            print(f"Pedestrian with id {pedestrian_id} is added.")
        elif switch == "a":    
            print(f"{pedestrians_number} pedestrians added.")

    def delete(self, scenario):
        """
        Delete a dynamic pedestrian with specific id
        """
        self.get_info(scenario)
        self.primary = False
        id = input(f"Enter pedestrian id from {self.id} ")
        if int(id) in self.id:
            print("\n------------------------- Deleting the pedestrian:\n")
            for i in range(len(self.dynamicElements)):
                if self.dynamicElements[i]["attributes"]["id"] == int(id):
                    index = i
            del self.dynamicElements[index]
        else:
            print("Exited: Wrong id!")
        
        self.scenario_dict["scenario"]["topography"]["dynamicElements"] = self.dynamicElements
        scenario.write_scenario(self.scenario_dict)
        

    def edit(self, scenario):
        """
        Edit Pedestrian attributes.
        TODO: Implement this function
        """
        self.get_info(scenario)
        self.primary = False
        print("\n-------------------------- Editing the pedestrian:\n")
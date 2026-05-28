import random

"""
Store and update device energy
"""
class ClientEnergy:
    def __init__(self, device_type):
        self.device_type = device_type
        self.curr_battery_level = 0
        self.battery_used = 0
        self.training_time = 0
        self.initialise_energy()
    
    def get_device_type(self):
        return self.device_type

    def get_battery(self):
        return self.curr_battery_level
    
    def get_used_battery(self):
        return self.battery_used
    
    def get_training_time(self):
        return self.training_time
    
    def get_initial_battery(self):
        return self.init_battery_level

    def initialise_energy(self):
        """
        Determine the initial device energy in Wh
        Assumes a 5 V battery and 3000 mAh capacity
        Assign the training times based on profiles from Cho et al.
        https://dl.acm.org/doi/10.1145/3550289
        """

        if (self.device_type=='H'):
            perc = random.randint(50, 90)
            self.curr_battery_level= (3000 * (perc/100) * 5) / 1000
            self.training_time= 16.11 + random.randint(-5, 5)

        if (self.device_type == 'M'):
            perc = random.randint(50, 90)
            self.curr_battery_level = (3000 * (perc/100) * 5) / 1000
            self.training_time = 50.31 + random.randint(-5, 5)

        if(self.device_type == 'L'):
            perc = random.randint(50, 90)
            self.curr_battery_level = (3000 * (perc/100) * 5) / 1000
            self.training_time = 38.18 + random.randint(-5, 5)
        
        self.init_battery_level = self.curr_battery_level
    
    def update_energy(self):
        """
        Update the current device energy based on profiles from
        by Cho et al.
        https://dl.acm.org/doi/10.1145/3550289

        Returns:
            Total amount of energy used by device 
        """

        if (self.device_type == 'H'):
            energy = 15.5 / 3600 # convert from J to Wh
        if (self.device_type == 'M'):
            energy = 27.3 / 3600 # convert from J to Wh
        if (self.device_type == 'L'):
            energy = 69.9 / 3600 # convert from J to Wh
            
        self.curr_battery_level -= energy
        self.battery_used += energy

        return self.battery_used

    def __str__(self):
        return f"Used energy: {self.battery_used:.2f}"
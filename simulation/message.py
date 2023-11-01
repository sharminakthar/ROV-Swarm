class Message:

    def __init__(self, drone_id:int, heading:float, speed:float):
        self.__drone_id = drone_id
        self.__heading = heading
        self.__speed = speed

    def get_drone_id(self):
        return self.__drone_id

    def get_heading(self):
        return self.__heading

    def get_speed(self):
        return self.__speed
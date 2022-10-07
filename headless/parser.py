import argparse
from enum import Enum
from flock_settings import Setting


class Parser():
    def __init__(self, settings):
        self.parser: argparse.ArgumentParser = self.parser_factory(settings)

    def parser_factory(self, settings):
        """Build a parser that can accept all of the configured parameters"""
        parser = argparse.ArgumentParser()

        parser.add_argument("--steps", type=int, required=True, dest="STEPS")
        parser.add_argument("--samples", type=int, required=False, dest="SAMPLES", default=1)
        
        for (name, setting) in Setting.__members__.items():
            default = settings.get_default(setting)
            type_of = type(default)
            argument = name.lower()

            if(type_of is int or type_of is float):

                parser.add_argument('--'+argument, type=type_of, default=default, dest=name)

            if(isinstance(default, Enum)):
                parser.add_argument("--"+argument, type=str, default=default, dest=name, choices=list(type_of._member_names_))

        return parser

    def parse_cmd_arguments(self):
        """Return a dictionary of parsed command line arguments"""
        return vars(self.parser.parse_args())

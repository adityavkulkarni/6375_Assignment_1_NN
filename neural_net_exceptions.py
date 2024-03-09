class NeuronInputException(Exception):
    def __init__(self, message):
        super().__init__(message)
        print(f"EXCEPTION | {message}")

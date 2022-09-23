from result_manager import ResultManager

class Test():
    def __init__(self):
        self.name = 'Niklas'
        self.value = 42

    def getName(self):
        return self.name

if __name__ == '__main__':
    test = Test()
    test.name = 'Niklas'
    resultManager = ResultManager()

    # resultManager.save_result(result=test, filename='test.pkl')
    result = resultManager.load_result(filename='test.pkl')
    print(result.getName())
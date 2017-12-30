import numpy as np
import time

class Test():
    def __init__(self):
        self.test = np.random.rand(32,32, 84, 84)


    def foo1(self):
        return self.test[1,:,:,:]

    def foo2(self):
        return self.test[1,...]

tester = Test()
ntests = int(5e6)



start = time.time()

for _ in range(0, ntests):

    test2 = tester.foo2()

end = time.time()

print(end - start)


start = time.time()

for _ in range(0, ntests):

    test1 = tester.foo1()

end = time.time()

print(end - start)

np.testing.assert_array_equal(test1, test2)
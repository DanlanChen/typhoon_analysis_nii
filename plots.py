#1 plot prediction and actual value
import matplotlib.pyplot as plt
f3 =  "small_cnn_output.jpg"
y_predict =[19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844, 19.749351501464844]
y_test = [1002.0, 994.0, 975.0, 900.0, 900.0, 900.0, 940.0, 900.0, 985.0, 925.0, 900.0, 985.0, 992.0, 925.0, 940.0, 1000.0, 920.0, 900.0, 980.0]
plt.title("test prediction and actual test data")
plt.plot(y_predict[:100],'g^', label = 'predict')
plt.plot(y_test[:100], 'r--',label = 'actual')
plt.legend(loc = 'upper left', shadow = True)
plt.savefig(f3)
plt.close()
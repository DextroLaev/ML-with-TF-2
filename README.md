# ML-with-TF-2

### Before proceeding forward make sure you have the installed the following libraries:

    Tensorflow 2.4.1
    Numpy 1.19.5
    Matplotlib 3.3.4
    Pandas 1.2.3
    sklearn 0.24.2
    
###  Installation process :     
  
      $ pip3 install tensorflow
      $ pip3 install pandas
      $ pip3 install numpy
      $ pip3 install matplotlib
      $ pip3 install sklearn
      
### Linear Regression  
      $ python3 linear_regression.py
### Logistic Regression      
  Note that, in the logistic regression we used roc graph to find out the threshold value, but as the 
  data is balanced data you can use threshold of 0.5 for better performance.
      
      $ python3 logistic.py
### Nelder-Mead optimization method
   An optimization algorithm to find minimum of an objective function in a multidimensional space without calculating gradient.
   In the given file , I have used (Himmelblau's function) and (Rosenbrock banana function).
   
   Rosenbrock banana function : https://en.wikipedia.org/wiki/Rosenbrock_function.
   
   Himmelblau's function : https://en.wikipedia.org/wiki/Himmelblau%27s_function.
   
   Nelder-Mead method : https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method.
        
       $ python3 nelder_mead_optimizer.py

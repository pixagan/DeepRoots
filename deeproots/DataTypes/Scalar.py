##Created : Anil Variyar
#last Modified : Anil Variyar | 12-02-2025
#Copyright : Anil Variyar | 2024 - 
#Terms of Use : MIT License | Read Terms and Conditions in the License file 
#No warranty of any kind is provided. Use at your own risk.

#A variable with a value, gradient, operation, parents and a backward function
# Setup to allow for automatic differentiation

class Scalar:

    def __init__(self, value, _parents=(), label='',operation=''):
        self.value     = value
        self.gradient  = 0.0
        self.operation = ''
        self.parents = set(_parents)
        self._backward = lambda *args: print("Backward Leaf : ",self.label)
        self.label = label

    def __repr__(self):
        return f"{self.label} | {self.value}"
    
    def __add__(self, other):
        print(self.label, print("+"))

        out = None
        
        if(isinstance(other, Scalar)):
            sum = self.value + other.value
            out =  Scalar(sum, (self, other),'+') #storing reference to self and other
            
            def _backward():
                print("Backward : ", self.label)

                self.gradient  += 1.0*out.gradient
                other.gradient += 1.0*out.gradient

                self._backward()
                other._backward()
            
            out._backward = _backward

            return out

        
        elif (isinstance(other, int) or isinstance(other, float)):
            sum = self.value + other
            out =  Scalar(sum, (self, Scalar(other)), '+')

            def _backward():
                print("Backward : ", self.label)

                self.gradient  += 1.0*out.gradient

                self._backward()
            
            out._backward = _backward
            
            return out
        else:
            raise TypeError("Invalid Operand for Addition of the Scalar datatype")



    def __radd__(self, other):
        return self+other
    
    
    def __mul__(self, other):
        print(self.label, print("*"))
        if(isinstance(other, Scalar)):
            product = self.value * other.value

            out =  Scalar(product, (self, other), '*') #storing reference to self and other

            def _backward():
                print("Backward : ", self.label)

                self.gradient  += other.value*out.gradient
                other.gradient += self.value*out.gradient

                self._backward()
                other._backward()
            
            out._backward = _backward
            return out
            
        elif (isinstance(other, int) or isinstance(other, float)):
            print("Multiplication Scalar ", self)
            product = self.value * other
            out =  Scalar(product, (self, Scalar(other)), '*')

            def _backward():
                print("Backward C: ", self.label)

                self.gradient  += other*out.gradient
                self._backward()
            
            out._backward = _backward

            
            return out


        
        else:
            raise TypeError("Invalid Operand for Multiplication of the Scalar datatype")




    def __rmul__(self, other):
        return self*other

    
    
    def get_parents(self):
        print(self.parents)

    

    def clear_gradient(self):
        self.gradient = 0.0
                
    


#----------------------------------------------------------------------



def main():

    a = Scalar(10)
    b = Scalar(20)

    c = a+b

    print(c)


#----------------------------------------------------------------------

if __name__ == "__main__":
    main()


#----------------------------------------------------------------------

        

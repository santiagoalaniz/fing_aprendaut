class CampoContinuo:
    # Constructor
    def __init__(self, nombre):
        self.nombre = nombre

    def rangos(self, max, mean, min):
        self.alto = (max + mean) / 2
        self.medio = mean
        self.bajo = (mean + min) / 2

    def calificar(self, valor):
        if valor >= self.alto:
            return 3
        elif valor >= self.medio:
            return 2
        elif valor >= self.bajo:
            return 1
        else:
            return 0
        
    def print(self):
        print(self.nombre)
        print("Alto: ( +inf ,", str(self.alto), "]")
        print("Medio Alto: (", str(self.alto) , ",", str(self.medio), "]")
        print("Medio Bajo: (", str(self.medio) , ",", str(self.bajo), "]")
        print("Bajo: (", str(self.bajo) , ", -inf)")
        print() 

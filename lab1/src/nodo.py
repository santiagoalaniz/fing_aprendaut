class Nodo:
    hoja = "hoja"
    rama = "rama"

    def __init__(self, attr, valor, tipo: str = rama, hijos: list['Nodo'] = None, resultado: str = None):
        if tipo not in [Nodo.hoja, Nodo.rama]:
            raise ValueError("Tipo must be 'hoja' or 'rama'")
        
        if tipo == Nodo.hoja and resultado is None:
            raise ValueError("Resultado must be provided if tipo is 'hoja'")
        
        if tipo == Nodo.hoja and hijos is not None:
            raise ValueError("Hijos must not be provided if tipo is 'hoja'")
        
        if tipo == Nodo.rama and resultado is not None:
            raise ValueError("Resultado must not be provided if tipo is 'rama'")
        
        if tipo == Nodo.rama and hijos is None:
            raise ValueError("Hijos must be provided if tipo is 'rama'")
        
        self.attr = attr
        self.valor = valor
        self.tipo = tipo
        self.hijos = hijos
        self.resultado = resultado


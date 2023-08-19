class Nodo:
    hoja = "hoja"
    rama = "rama"

    def __init__(self, attr, valor, tipo: str = rama, hijos: list['Nodo'] = None, resultado: str = None, n_casos: int = None):
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
        
        if n_casos is None or n_casos < 0:
            raise ValueError("n_casos must be greater than 0")
        
        self.attr = attr
        self.valor = valor
        self.tipo = tipo
        self.hijos = hijos
        self.resultado = resultado
        self.n_casos = n_casos

    def __str__(self):
        return f"{self.tipo} {self.attr} = {self.valor} -> {self.resultado} ({self.n_casos})"

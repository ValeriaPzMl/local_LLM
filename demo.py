def saludo(nombre):
    return f"Hola, {nombre}"


def suma(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Ambos parámetros deben ser números")
    return a + b
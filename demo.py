def saludo(nombre=None):
    if nombre is None:
        return "hola"
    else:
        return f"Hola, {nombre}"

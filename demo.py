def saludo(nombre=None):
    if nombre is None:
        return "hola"
    else:
        return f"Hola, {nombre}"


def suma(a, b):
    """Calcula la suma de dos números."""
    return a + b


if __name__ == "__main__":
    # Verificar compilación ejecutando el módulo
    print("Módulo demo.py compilado correctamente.")
    
    # Prueba básica de la función suma
    resultado = suma(3, 5)
    print(f"suma(3, 5) = {resultado}")
import re


def clean_text(text: str) -> str:
    """
    Limpia espacios innecesarios sin eliminar los saltos
    que separan párrafos.
    """
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Reduce espacios y tabulaciones repetidos.
    text = re.sub(r"[ \t]+", " ", text)

    # Evita demasiados saltos de línea seguidos.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[str]:
    """
    Divide un texto en fragmentos con una pequeña superposición.

    chunk_size:
        Cantidad aproximada de caracteres por fragmento.

    overlap:
        Cantidad de caracteres compartidos entre fragmentos.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que cero.")

    if overlap < 0:
        raise ValueError("overlap no puede ser negativo.")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap debe ser menor que chunk_size."
        )

    text = clean_text(text)

    if not text:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        candidate = (
            f"{current_chunk}\n\n{paragraph}".strip()
            if current_chunk
            else paragraph
        )

        if len(candidate) <= chunk_size:
            current_chunk = candidate
            continue

        if current_chunk:
            chunks.append(current_chunk)

            previous_tail = current_chunk[-overlap:]
            current_chunk = (
                f"{previous_tail}\n\n{paragraph}".strip()
            )
        else:
            # Si un solo párrafo es demasiado grande,
            # lo divide directamente.
            start = 0

            while start < len(paragraph):
                end = start + chunk_size
                chunks.append(paragraph[start:end].strip())
                start += chunk_size - overlap

            current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk)

    return [
        chunk
        for chunk in chunks
        if chunk
    ]
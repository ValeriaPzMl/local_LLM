from pathlib import Path

from pypdf import PdfReader


class DocumentLoader:
    @staticmethod
    def load(path: str) -> str:
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"No existe el archivo: {file_path}"
            )

        extension = file_path.suffix.lower()

        if extension == ".pdf":
            return DocumentLoader.load_pdf(file_path)

        if extension in {".txt", ".md"}:
            return DocumentLoader.load_text(file_path)

        raise ValueError(
            f"Tipo de archivo no compatible: {extension}"
        )

    @staticmethod
    def load_pdf(path: Path) -> str:
        reader = PdfReader(path)
        pages = []

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                pages.append(page_text)

        return "\n\n".join(pages)

    @staticmethod
    def load_text(path: Path) -> str:
        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )
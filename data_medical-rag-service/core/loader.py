from pypdf import PdfReader


def load_pdf(path):
    try:
        reader = PdfReader(path)
    except Exception as e:
        try:
            from pypdf.errors import DependencyError
        except Exception:
            DependencyError = None

        if DependencyError and isinstance(e, DependencyError):
            print(f" Skipping '{path}': cryptography>=3.1 required for encrypted PDFs.")
        else:
            print(f" Skipping '{path}': failed to open PDF ({e}).")

        return ""

    text = ""
    for page in reader.pages:
        try:
            page_text = page.extract_text()
        except Exception:
            page_text = None

        if page_text:
            text += page_text

    return text
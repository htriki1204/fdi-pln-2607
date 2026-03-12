#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "typer>=0.16.0",
# ]
# ///

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import typer

ASCII_OFFSET = 45


# Cuando leemos el fichero binario y deshacemos el desplazamiento de 45,
# obtenemos un "alfabeto intermedio" PLNCG26. Este diccionario contiene los transformadores
# directos: un byte de ese alfabeto se convierte siempre en el mismo
# carácter Unicode, así que aquí sí compensa usar una tabla directa.
DIRECT_STAGE_TO_TEXT = {
    ord("8"): " ",
    ord("7"): "\n",
    ord("s"): ".",
    ord("t"): ",",
    ord("u"): ";",
    ord("v"): ":",
    ord("~"): "(",
    0x7F: ")",
    ord("}"): "'",
    ord("|"): "'",
    0x91: '"',
    0x92: '"',
}

# Este es el camino contrario para encode. Guardamos la inversa de los
# símbolos que se pueden transformar de forma 1 a 1

DIRECT_TEXT_TO_STAGE = {
    " ": b"8",
    "\n": b"7",
    ".": b"s",
    ",": b"t",
    ";": b"u",
    ":": b"v",
    "(": b"~",
    ")": b"\x7f",
    "'": b"|",
}
# En PLNCG26 los números no aparecen como '0'..'9', sino como letras
# concretas del alfabeto intermedio. Por eso los separamos en otra tabla:
# al decodificar, por ejemplo, 'j' significa '1', y al codificar hacemos
# la transformación inversa.

STAGE_DIGIT_TO_TEXT = {
    ord("i"): "0",
    ord("j"): "1",
    ord("k"): "2",
    ord("l"): "3",
    ord("m"): "4",
    ord("n"): "5",
    ord("o"): "6",
    ord("p"): "7",
    ord("q"): "8",
    ord("r"): "9",
}

TEXT_DIGIT_TO_STAGE = {
    value: bytes([key]) for key, value in STAGE_DIGIT_TO_TEXT.items()
}

# Estos diccionarios representan modificadores. Aquí el byte no crea un
# carácter nuevo, sino que modifica el último carácter que ya habíamos
# producido. Por ejemplo, '_' convierte una vocal en su versión con tilde.
ACUTE = {
    "a": "á",
    "e": "é",
    "i": "í",
    "o": "ó",
    "u": "ú",
    "A": "Á",
    "E": "É",
    "I": "Í",
    "O": "Ó",
    "U": "Ú",
}

DIAERESIS = {
    "u": "ü",
    "U": "Ü",
}

TILDE = {
    "n": "ñ",
    "N": "Ñ",
}

ACCENTED_TO_BASE_AND_MODIFIER = {
    "á": ("a", b"_"),
    "é": ("e", b"_"),
    "í": ("i", b"_"),
    "ó": ("o", b"_"),
    "ú": ("u", b"_"),
    "ü": ("u", b"`"),
    "ñ": ("n", b"a"),
}

# Las letras latinas base se codifican como A-Z en el alfabeto intermedio.
# Esta tabla se usa en encode para pasar de una letra Unicode normal
# ('a'..'z') a su byte base dentro de PLNCG26.
LETTER_TO_STAGE = {
    chr(code): bytes([ord(chr(code).upper())]) for code in range(ord("a"), ord("z") + 1)
}


RECOGNIZED_STAGE_BYTES = (
    set(DIRECT_STAGE_TO_TEXT)
    | set(STAGE_DIGIT_TO_TEXT)
    | set(range(ord("A"), ord("Z") + 1))
    | {ord("_"), ord("`"), ord("a"), ord("b"), ord("{")}
)


def raw_to_stage(blob: bytes) -> bytes:
    """Deshace el desplazamiento del binario PLNCG26 para recuperar el alfabeto intermedio."""
    return bytes((byte + ASCII_OFFSET) % 256 for byte in blob)


def stage_to_raw(stage: bytes) -> bytes:
    """Aplica el desplazamiento inverso para convertir el alfabeto intermedio en binario PLNCG26."""
    return bytes((byte - ASCII_OFFSET) % 256 for byte in stage)


def decode_stage(stage: bytes) -> str:
    """Convierte bytes del alfabeto intermedio PLNCG26 en texto Unicode."""
    out: list[str] = []
    paren_toggle_open = True
    i = 0

    while i < len(stage):
        byte = stage[i]

        # '{{' no es un símbolo de un solo byte, sino un token de dos bytes. necesitamos mirar el byte actual y el siguiente al mismo tiempo.
        if byte == ord("{") and stage[i : i + 2] == b"{{":
            out.append("(" if paren_toggle_open else ")")
            paren_toggle_open = not paren_toggle_open
            i += 2
            continue

        direct = DIRECT_STAGE_TO_TEXT.get(byte)
        if direct is not None:
            out.append(direct)
        elif byte == ord("b"):
            # En PLNCG26 la mayúscula se marca después de la letra base. Es decir, primero sale la letra y luego 'b' la convierte en mayúscula.
            if out and out[-1].isalpha():
                out[-1] = out[-1].upper()
        elif byte == ord("_"):
            if out:
                out[-1] = ACUTE.get(out[-1], out[-1])
        elif byte == ord("`"):
            if out:
                out[-1] = DIAERESIS.get(out[-1], out[-1])
        elif byte == ord("a"):
            if out:
                out[-1] = TILDE.get(out[-1], out[-1])
        elif byte in STAGE_DIGIT_TO_TEXT:
            out.append(STAGE_DIGIT_TO_TEXT[byte])
        elif ord("A") <= byte <= ord("Z"):
            out.append(chr(byte + 32))
        else:
            out.append(bytes([byte]).decode("latin-1"))

        i += 1

    return "".join(out)


def decode_blob(blob: bytes) -> str:
    """Decodifica un fichero PLNCG26 completo desde bytes crudos hasta texto UTF-8."""
    return decode_stage(raw_to_stage(blob))


@dataclass
class EncodeState:
    """Mantiene el estado necesario durante la codificación carácter a carácter."""

    double_quote_open: bool = True


def encode_character(char: str, state: EncodeState) -> bytes:
    """Codifica un carácter Unicode como bytes del alfabeto intermedio PLNCG26."""
    direct = DIRECT_TEXT_TO_STAGE.get(char)
    if direct is not None:
        return direct

    digit = TEXT_DIGIT_TO_STAGE.get(char)
    if digit is not None:
        return digit

    if char == '"':
        token = b"\x91" if state.double_quote_open else b"\x92"
        state.double_quote_open = not state.double_quote_open
        return token

    lowercase = char.lower()
    modifier = b""
    uppercase = char.isalpha() and char == char.upper() and char != lowercase

    if lowercase in ACCENTED_TO_BASE_AND_MODIFIER:
        # Las vocales acentuadas, la ü y la ñ no tienen un byte independiente:
        # se representan como letra base + modificador. Por ejemplo, 'á' se
        # codifica como la base de 'a' seguida de '_'.
        lowercase, modifier = ACCENTED_TO_BASE_AND_MODIFIER[lowercase]

    if lowercase in LETTER_TO_STAGE:
        token = LETTER_TO_STAGE[lowercase] + modifier
        if uppercase:
            # Si la letra original era mayúscula, añadimos el modificador 'b'
            # al final. Por ejemplo, 'E' se codifica como la base de 'e' + 'b'.
            token += b"b"
        return token

    if (
        len(char) == 1
        and ord(char) < 256
        and not char.isalpha()
        and not char.isdigit()
        and ord(char) not in RECOGNIZED_STAGE_BYTES
    ):
        return bytes([ord(char)])

    raise ValueError(f"Caracter no soportado por PLNCG26: {char!r}")


def encode_text(text: str) -> bytes:
    """Codifica texto Unicode completo y devuelve el binario PLNCG26 final."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    state = EncodeState()
    stage = bytearray()

    for char in normalized:
        stage.extend(encode_character(char, state))

    return stage_to_raw(bytes(stage))


def detect_probability(blob: bytes) -> float:
    """Estima la probabilidad de que un blob binario pertenezca al plano PLNCG26."""
    if not blob:
        return 0.0

    # transformamos el binario al alfabeto intermedio
    stage = raw_to_stage(blob)
    recognized = 0

    for byte in stage:
        # cuenta cuantos bytes pertenecen a la codiciación
        # PLNCG26: letras base, digitos codificados, modificadores y signos
        # reservados del formato.
        if byte in RECOGNIZED_STAGE_BYTES:
            recognized += 1

    recognized_ratio = recognized / len(stage)
    return round(recognized_ratio, 4)


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Codificador y decodificador UTF-8 <-> PLNCG26.",
)


@app.command()
def decode(fichero: Path) -> None:
    """Convierte un fichero PLNCG26 a texto UTF-8 por stdout."""

    sys.stdout.write(decode_blob(fichero.read_bytes()))


@app.command()
def encode(fichero: Path) -> None:
    """Convierte un fichero UTF-8 a PLNCG26 por stdout."""

    text = fichero.read_text(encoding="utf-8")
    sys.stdout.buffer.write(encode_text(text))


@app.command()
def detect(fichero: Path) -> None:
    """Calcula una probabilidad aproximada de que el fichero sea PLNCG26."""

    probability = detect_probability(fichero.read_bytes())
    typer.echo(f"{probability:.2%}")


if __name__ == "__main__":
    app()

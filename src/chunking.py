"""Two chunking strategies: A (naive fixed-size) and B (structure-aware)."""
import re


# ---------------------------------------------------------------- Strategy A
def naive_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Fixed-size character windows. Ignores sentences, lists, tables, headers."""
    step = chunk_size - overlap
    return [text[i:i + chunk_size] for i in range(0, len(text), step)
            if text[i:i + chunk_size].strip()]


# ---------------------------------------------------------------- Strategy B
def _split_sections(text: str):
    """Yield (breadcrumb, body) for every markdown header section."""
    path, body, out = {}, [], []

    def flush():
        content = "\n".join(body).strip()
        if content:
            crumb = " > ".join(path[k] for k in sorted(path))
            out.append((crumb, content))
        body.clear()

    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            path = {k: v for k, v in path.items() if k < level}
            path[level] = m.group(2).strip()
        else:
            body.append(line)
    flush()
    return out


def _blocks(body: str) -> list[str]:
    """Paragraph-level blocks. A paragraph ending in ':' is glued to the list/table
    that follows it, so 'Common risk factors include:' stays with its bullets."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    blocks, i = [], 0
    while i < len(paras):
        cur = paras[i]
        while cur.endswith(":") and i + 1 < len(paras):
            i += 1
            cur += "\n" + paras[i]
        blocks.append(cur)
        i += 1
    return blocks


def _split_oversized(block: str, limit: int) -> list[str]:
    """Fallback for a block bigger than `limit`: split on lines, then sentences."""
    pieces = []
    for line in block.split("\n"):
        if len(line) <= limit:
            pieces.append(line)
        else:
            pieces += re.split(r"(?<=[.!?])\s+", line)
    return pieces


def structure_chunks(text: str, max_chars: int = 1000) -> list[str]:
    """Header -> paragraph/list/table -> line/sentence splitting.
    Every chunk is prefixed with its section breadcrumb for context."""
    chunks = []
    for crumb, body in _split_sections(text):
        units = []
        for b in _blocks(body):
            units += [b] if len(b) <= max_chars else _split_oversized(b, max_chars)
        cur = ""
        for u in units:
            if cur and len(cur) + len(u) + 1 > max_chars:
                chunks.append(f"[{crumb}]\n{cur}")
                cur = ""
            cur = f"{cur}\n{u}".strip() if cur else u
        if cur:
            chunks.append(f"[{crumb}]\n{cur}")
    return chunks

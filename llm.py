import re

MAX_MAP_CHUNKS = 6
MAX_QUIZ_CARDS = 4
MAX_CONTEXT_CHARS = 1400


def sample_evenly(items, n):
    if not items:
        return []
    if len(items) <= n:
        return list(items)
    indexes = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)]
    seen = set()
    sampled = []
    for idx in indexes:
        if idx not in seen:
            seen.add(idx)
            sampled.append(items[idx])
    return sampled


def generate(qa_pipe, prompt, max_new_tokens=200):
    result = qa_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        truncation=True,
    )
    return result[0]["generated_text"].strip()


def _chunk_text(chunk):
    return chunk["text"] if isinstance(chunk, dict) else chunk


def summarize_notes(qa_pipe, chunks, on_progress=None):
    sampled = sample_evenly(chunks, MAX_MAP_CHUNKS)
    total = len(sampled) + 1
    partials = []
    for i, chunk in enumerate(sampled, start=1):
        if on_progress:
            on_progress(i, total, f"Summarizing section {i} of {len(sampled)}")
        notes = _chunk_text(chunk)[:MAX_CONTEXT_CHARS]
        prompt = (
            "Write a concise study summary of these notes. "
            "Cover key concepts, definitions, and important facts.\n\n"
            f"Notes:\n{notes}"
        )
        partials.append(generate(qa_pipe, prompt, max_new_tokens=180))

    if len(partials) == 1:
        if on_progress:
            on_progress(total, total, "Finishing summary")
        return partials[0]

    if on_progress:
        on_progress(total, total, "Combining section summaries")
    combined = "\n".join(f"- {part}" for part in partials)[:MAX_CONTEXT_CHARS]
    prompt = (
        "Combine these section summaries into one structured study recap "
        "with key concepts, definitions, and important facts.\n\n"
        f"{combined}"
    )
    return generate(qa_pipe, prompt, max_new_tokens=320)


def answer_question(qa_pipe, query, hits):
    if not hits:
        return "I don't see that in your notes."

    parts = []
    for hit in hits:
        label = f"[Page {hit['page']}]" if hit.get("page") else "[Notes]"
        parts.append(f"{label} {hit['text']}")
    context = "\n".join(parts)[:1800]
    prompt = (
        "Use only the context. If the answer is not in the context, "
        "reply exactly: I don't see that in your notes.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    return generate(qa_pipe, prompt, max_new_tokens=180)


def parse_qa(raw):
    question_match = re.search(r"Q:\s*(.+?)(?:\n\s*A:|$)", raw, re.S | re.I)
    answer_match = re.search(r"A:\s*(.+)$", raw, re.S | re.I)
    question = question_match.group(1).strip() if question_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    if not question:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        question = lines[0] if lines else raw.strip()
        answer = " ".join(lines[1:]) if len(lines) > 1 else ""
    return question, answer or "See the source notes."


def generate_quiz_cards(qa_pipe, chunks, n=MAX_QUIZ_CARDS, on_progress=None):
    sampled = sample_evenly(chunks, n)
    cards = []
    for i, chunk in enumerate(sampled, start=1):
        if on_progress:
            on_progress(i, len(sampled), f"Writing question {i} of {len(sampled)}")
        notes = _chunk_text(chunk)[:MAX_CONTEXT_CHARS]
        prompt = (
            "Create one short study quiz question and answer from the notes. "
            "Use this format:\nQ: ...\nA: ...\n\n"
            f"Notes:\n{notes}"
        )
        question, answer = parse_qa(generate(qa_pipe, prompt, max_new_tokens=160))
        if question:
            cards.append(
                {
                    "question": question,
                    "answer": answer,
                    "page": chunk.get("page") if isinstance(chunk, dict) else None,
                }
            )
    return cards

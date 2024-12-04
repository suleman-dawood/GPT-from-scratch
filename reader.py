def read_tokens():
    with open("vocab.txt", "r", encoding="utf-8-sig") as file:
        tokens = [line.strip() for line in file]
    return tokens

def justified_print(text, length_thr=120):
    lines = text.split('\n')

    for line in lines:
        words = line.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 > length_thr:
                print(current_line)
                current_line = word + " "
            else:
                current_line += word + " "

        if current_line:
            print(current_line.rstrip())
        else:
            print()
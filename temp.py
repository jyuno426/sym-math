# -*- coding: utf-8 -*-

if __name__ == '__main__':
    with open("dataset/local/pred-test.txt", "r") as f:
        with open("dataset/local/pred-test-mapping.txt", "r") as g:
            flines = f.readlines()
            glines = g.readlines()
            assert len(flines) == len(glines)
            n = len(flines)
            cnt = 0
            for i in range(n):
                if flines[i] == glines[i]:
                    cnt += 1

    print(cnt, n)
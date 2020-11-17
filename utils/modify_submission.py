#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-06-20 10:04
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import csv


def modify_submission():
    with open("submission.csv", "r") as submission_origin:
        f_origin = csv.reader(submission_origin)
        headers = next(f_origin)
        new_rows = []
        for row in f_origin:
            # print(row)
            new_score = str(1 - float(row[1]))
            new_row = [row[0], new_score]
            new_rows.append(new_row)

    with open('submission_modified.csv', 'w') as submission_modified:
        f_modified = csv.writer(submission_modified)
        f_modified.writerow(headers)
        for row in new_rows:
            f_modified.writerow(row)


def main():
    """
    docstring
    """
    pass


if __name__ == "__main__":
    main()

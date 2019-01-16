// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <string.h>


typedef struct {
    std::string label;
    bool useclass;
} labelinfo;


// read labels from file, combine with useclass flags to return a vector
// with label name and whether it is used
std::vector <labelinfo> readlabels(std::string labelfile, std::string useclassflags) {
    // ----------------
    // Read class names
    // ----------------
    std::vector<labelinfo> labels;

    const int nuseclasses = useclassflags.length();

    char *classflags = NULL;
    classflags = static_cast<char *>(malloc(nuseclasses));

    if (classflags == NULL) {
        std::cout << "failed to allocate memory for class label" << std::endl;
    } else {
        memcpy(classflags, useclassflags.c_str(), nuseclasses);

        std::ifstream infile(labelfile, std::ifstream::in);

        if (!infile.is_open()) {
            std::cout << "Could not open labels file" << std::endl;
        } else {
            std::string tmpstr;
            while (infile.good()) {
                getline(infile, tmpstr);
                if (infile.good()) {
                    labelinfo tmplabel;
                    tmplabel.label = tmpstr;
                    int pos = labels.size();
                    if (pos < nuseclasses) {
                        tmplabel.useclass=(classflags[pos] == '1')?true:false;
                    } else {
                        tmplabel.useclass = false;
                    }
                    labels.push_back(tmplabel);
                }
            }
            infile.close();
        }

        if (classflags != NULL)
            free(classflags);
    }

    return labels;
}

# if 0
int main() {
    std::string useclassflags = "00000010000000100";
    std::string labelfile = "pascal_voc_classes.txt";
    std::vector<labelinfo> labels = readlabels(labelfile, useclassflags);
    for (int i = 0; i < labels.size(); i++) {
        printf("%s %d\n", labels[i].label.c_str(), labels[i].useclass);
    }
}
#endif
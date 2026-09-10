# Third-party notices

This extension's own code is licensed under MIT. Model weights and referenced projects have their own terms.

Referencing or fetching upstream code does not grant permission to use, modify, or redistribute it. The licensing of the complete dependency stack has not been cleared. The pinned HPSv3-PlusPlus commit is `6a095f68ee98330bf22365f872ed609bd44a216f`; the hpsv3-4bit commit is `f9878d0535205da70fe701e6a4df2abd268b88d7`.

- [HPSv3-PlusPlus](https://github.com/PlantPotatoOnMoon/HPSv3-PlusPlus) is referenced through the nested Git submodule in hpsv3-4bit. The pinned upstream checkout has no LICENSE file. This extension does not copy its implementation or include it in release archives; users retrieve it directly from upstream. This extension's MIT license does not cover that code.
- The Hugging Face TRL compatibility shim in hpsv3-4bit is licensed under Apache-2.0. Its [attribution](https://github.com/Stella2211/hpsv3-4bit/blob/main/THIRD_PARTY_NOTICES.md) and [license text](https://github.com/Stella2211/hpsv3-4bit/blob/main/licenses/TRL-Apache-2.0.txt) remain in that repository.
- [HPSv3-PlusPlus-bnb-NF4](https://huggingface.co/stella221125/HPSv3-PlusPlus-bnb-NF4) weights are downloaded separately. Refer to the model repository's LICENSE and NOTICE. Weight licensing does not establish licensing for the upstream Python implementation.

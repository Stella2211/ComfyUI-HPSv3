# Third-party notices

This extension's own code is licensed under MIT. Model weights and referenced projects have their own terms.

Referencing or fetching upstream code does not grant permission to use, modify, or redistribute it. The licensing of the complete dependency stack has not been cleared. The pinned HPSv3-PlusPlus commit is `6a095f68ee98330bf22365f872ed609bd44a216f`; the hpsv3-4bit commit is `650f86cc1830463b5bd8af073da27c04becb1fce`.

- [HPSv3-PlusPlus](https://github.com/PlantPotatoOnMoon/HPSv3-PlusPlus) is referenced through the nested Git submodule in hpsv3-4bit. The pinned upstream checkout has no LICENSE file. This extension does not copy its implementation or include it in release archives; users retrieve it directly from upstream. This extension's MIT license does not cover that code.
- The Hugging Face TRL compatibility shim in hpsv3-4bit is licensed under Apache-2.0. Its [attribution](https://github.com/Stella2211/hpsv3-4bit/blob/main/THIRD_PARTY_NOTICES.md) and [license text](https://github.com/Stella2211/hpsv3-4bit/blob/main/licenses/TRL-Apache-2.0.txt) remain in that repository.
- [HPSv3-PlusPlus-bnb-NF4](https://huggingface.co/stella221125/HPSv3-PlusPlus-bnb-NF4) weights are downloaded separately. Refer to the model repository's LICENSE and NOTICE. Weight licensing does not establish licensing for the upstream Python implementation.
- HPSv3 uses the same pinned hpsv3-4bit repository's `hpsv3` wrapper. Its model class is adapted from MizzenAI/HPSv3 under MIT; the upstream attribution and license text are retained in that dependency's THIRD_PARTY_NOTICES.md. The wrapper is fetched during installation and excluded from this extension's release archive.
- [HPSv3-bnb-NF4](https://huggingface.co/stella221125/HPSv3-bnb-NF4) weights are downloaded separately. Refer to that model repository's license and notices for the weight terms.

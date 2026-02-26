# Changelog

## [2.0.0](https://github.com/nicokim/indextts2-inference/compare/v1.0.0...v2.0.0) (2026-02-26)


### ⚠ BREAKING CHANGES

* de-vendor transformers, support transformers 5.x ([#2](https://github.com/nicokim/indextts2-inference/issues/2))
* **webui:** Easier DeepSpeed launch argument

### Features

* **accel:** add batch support ([44b5ffc](https://github.com/nicokim/indextts2-inference/commit/44b5ffcb5dda05217e08521d58a919a44c6dac2b))
* achieve inference acceleration for the gpt2 stage ([c1ef414](https://github.com/nicokim/indextts2-inference/commit/c1ef4148af34dcb1943ae1fe4bded0f7602f68bb))
* achieve inference acceleration for the gpt2 stage (3.79×) ([1d5d079](https://github.com/nicokim/indextts2-inference/commit/1d5d079aaa290a7179967e80b7e25f5aa002cd8f))
* achieve inference acceleration for the s2mel stage (1.61×) ([e42480c](https://github.com/nicokim/indextts2-inference/commit/e42480ced8af8bf8ef596b2420faa5c86fa64ed2))
* add batch support ([3c36027](https://github.com/nicokim/indextts2-inference/commit/3c360273da9738158ee72bf6d47710033f835049))
* Add reusable Emotion Vector normalization helper ([8aa8064](https://github.com/nicokim/indextts2-inference/commit/8aa8064a53c5b5b53ff20f6de94aaadc18a4cd9d))
* **cli:** Support XPU ([#322](https://github.com/nicokim/indextts2-inference/issues/322)) ([e83df4e](https://github.com/nicokim/indextts2-inference/commit/e83df4e4270e6b1f5f08b341907689c078a382c4))
* de-vendor transformers, support transformers 5.x ([#2](https://github.com/nicokim/indextts2-inference/issues/2)) ([ad522fe](https://github.com/nicokim/indextts2-inference/commit/ad522fe80dad9aefd87ce7c233d02b17f6e11733))
* DeepSpeed is now an optional dependency which can be disabled ([936e6ac](https://github.com/nicokim/indextts2-inference/commit/936e6ac4dd4558b29c431e709c57a04cbd9c78bc))
* Extend GPU Check utility to support more GPUs ([39a035d](https://github.com/nicokim/indextts2-inference/commit/39a035d1066dee3baeb45a5580555ea2013591f0))
* **front.py:** add regex pattern for technical terms ([82a5b90](https://github.com/nicokim/indextts2-inference/commit/82a5b9004a90bb6a9656cfb95e3aaa05d1117a0a))
* gumbel_softmax_sampler ([a3b884f](https://github.com/nicokim/indextts2-inference/commit/a3b884ff6ff401923e6242fb87630be0a6d0f221))
* **i18n:** Add missing UI translation strings ([5f0b0a9](https://github.com/nicokim/indextts2-inference/commit/5f0b0a9f9c57c05440d6d3992e90792d70023ae5))
* Implement `emo_alpha` scaling of emotion vectors and emotion text ([9668064](https://github.com/nicokim/indextts2-inference/commit/9668064377fcba7d3e8643c8680d5f8d37ddb2a9))
* **indextts:** add glossary support for custom term pronunciations ([6deed97](https://github.com/nicokim/indextts2-inference/commit/6deed97efe636bf68825ebb0ec82ec994fd1d42f))
* optimize s2mel stage ([31e7e85](https://github.com/nicokim/indextts2-inference/commit/31e7e855e21779c5ad0cd6b7b0ae4329c0c91ec2))
* **sampler:** enhance with greedy sampling mode ([42a7339](https://github.com/nicokim/indextts2-inference/commit/42a73394e9f9049699bc8e5210a705249e248b3f))
* Warn if input text contains UNK tokens ([34be9bf](https://github.com/nicokim/indextts2-inference/commit/34be9bfb146ed4cf41808f002fc47332ddddaebd))
* **webui.py:** add glossary term validation and error handling ([a7099c4](https://github.com/nicokim/indextts2-inference/commit/a7099c4a6e13a59b2b4e9c8e90e06b61595412f0))
* **webui:** Easier DeepSpeed launch argument ([f0badb1](https://github.com/nicokim/indextts2-inference/commit/f0badb13af7f84f92caa4d7d65b56ef4e3456337))
* **webui:** Implement emotion weighting for vectors and text modes ([d899770](https://github.com/nicokim/indextts2-inference/commit/d8997703137434e2d71329721e52e88eff534572))
* **webui:** Implement speech synthesis progress bar ([555e146](https://github.com/nicokim/indextts2-inference/commit/555e146fb4066d465225a850b21243e5b7eccd0e))
* 归一化参数到推荐的范围，改善用户体验 ([48a71af](https://github.com/nicokim/indextts2-inference/commit/48a71aff6da497c3d14c9a0bd2be9d627c8c45d3))
* 归一化参数到推荐的范围，改善用户体验 ([af2b06e](https://github.com/nicokim/indextts2-inference/commit/af2b06e061937d291e276a0aa1bd4a2254a06078))
* 裁剪过长的输入音频至15s,减少爆内存和显存 ([009428b](https://github.com/nicokim/indextts2-inference/commit/009428b62dc60b9d7bfdddd1641d51df9c2afa80))
* 裁剪过长的输入音频至15s,减少爆内存和显存 ([0828dcb](https://github.com/nicokim/indextts2-inference/commit/0828dcb098247760ba17ba5ad8a5b6b5cb460f95))


### Bug Fixes

* add force_rebuild flag for fused alias_free_activation and update installation instructions ([59c05c0](https://github.com/nicokim/indextts2-inference/commit/59c05c0765b61f4c8fe5d66ff7fe3c6a960e09e4))
* Add support for melancholic emotion in text-to-emotion vectors ([a6a955d](https://github.com/nicokim/indextts2-inference/commit/a6a955d2aa18d6400b2f4cdaad35e792d5bb231b))
* **cli:** More robust device priority checks ([6113567](https://github.com/nicokim/indextts2-inference/commit/6113567e94a885193f451d3d81665a609e9efbe5))
* Don't load DeepSpeed if use_deepspeed is False ([05a8ae4](https://github.com/nicokim/indextts2-inference/commit/05a8ae45e5da8ab7bdb035069181c11dfa3ac2bf))
* Empty generator -&gt; IndexError problem on non-streaming infer() ([db5b39b](https://github.com/nicokim/indextts2-inference/commit/db5b39bb6ad903c219b2dd33d60b0f0bdaede664))
* Empty generator -&gt; IndexError problem on non-streaming infer() ([750d9d9](https://github.com/nicokim/indextts2-inference/commit/750d9d9d15c90cfbf2d1f386a481e0087a834a61))
* Fast and robust text-to-emotion algorithm ([58ad225](https://github.com/nicokim/indextts2-inference/commit/58ad225fb41c99ebf873f3b1694391f8164ac0d3))
* Fix character encoding in examples ([55b7d32](https://github.com/nicokim/indextts2-inference/commit/55b7d321495b530c5467ecdf0e06fb996a011a71))
* Fix internal text-to-emotion vector labels ([feba501](https://github.com/nicokim/indextts2-inference/commit/feba5010137460ec8decaba2603da7819cb963e6))
* **front.py:** load full term glossary entries from yaml file ([1460adb](https://github.com/nicokim/indextts2-inference/commit/1460adbdc54d13af39b155e2760d0045ac23a825))
* handle multiple sentence placeholders in de_tokenized_by_CJK_char ([267e344](https://github.com/nicokim/indextts2-inference/commit/267e344a0956c09f4c2ff9e16d1c6bda7d67c11a))
* Improve .gitignore and re-add config file ([5ffb84b](https://github.com/nicokim/indextts2-inference/commit/5ffb84b427d2d7e8fa597cc4f5e31c2b30cfd492))
* **infer_v2:** Correct the import path of BigVGAN's custom cuda kernel ([e409c4a](https://github.com/nicokim/indextts2-inference/commit/e409c4a19b911f203de69f043987e9f37d78b0c8))
* Suppress pandas PyArrow future dependency warning ([d5cdb5e](https://github.com/nicokim/indextts2-inference/commit/d5cdb5eb3cfb34cbb901e4da4a0302c07e6a5e06))
* Update pandas to fix Gradio errors ([3e64c4a](https://github.com/nicokim/indextts2-inference/commit/3e64c4ac11c12d119567fa3a1b47da577170fbc8))
* update PINYIN_TONE_PATTERN and NormalizerZh ([7d943b3](https://github.com/nicokim/indextts2-inference/commit/7d943b362dbc950edc2dfc1fb7726dbfd5d39b81))
* use simple version tags (v*) for release-please ([#4](https://github.com/nicokim/indextts2-inference/issues/4)) ([76c70aa](https://github.com/nicokim/indextts2-inference/commit/76c70aa035a42013473cf5ad39328dba77cbcf7d))
* Use WeTextProcessing on Linux, and wetext on other platforms ([dcdb061](https://github.com/nicokim/indextts2-inference/commit/dcdb0614bf7fc29fe560969324478c29928b1695))
* **webui.py:** replace return statements with warnings and update markdown table ([fa7962f](https://github.com/nicokim/indextts2-inference/commit/fa7962f1f2decd7a430d2d62544d2e3a3e596d06))
* **webui.py:** strip tailing whitespace for glossary terms ([6b6606a](https://github.com/nicokim/indextts2-inference/commit/6b6606a2f4d1974ea8ae1413e609b7a17af882fc))
* **webui:** Add support for Gradio 5.45.0 and higher ([ef09710](https://github.com/nicokim/indextts2-inference/commit/ef097101b7ab70120250c9c3f6394174e656590e))
* **webui:** Experimental checkbox bugfixes and add visual warning label ([ec368de](https://github.com/nicokim/indextts2-inference/commit/ec368de9329c7beffa3df369005c7d25e7684980))
* **webui:** Fix unintentional empty spacing between control groups ([f041d8e](https://github.com/nicokim/indextts2-inference/commit/f041d8eb64e1af1f7a5def01247d564b1a283a5b))
* **webui:** Make the Advanced Settings visible by default again ([e185fa1](https://github.com/nicokim/indextts2-inference/commit/e185fa1ce748de6759f772f3852024c5443724f3))
* **webui:** Make the Emotion Control Weight slider visible again ([c5f9a31](https://github.com/nicokim/indextts2-inference/commit/c5f9a311275447707791c2758a2267dd137142f2))
* **webui:** New default emo_alpha recommendation instead of scaling ([1520d06](https://github.com/nicokim/indextts2-inference/commit/1520d0689baa4bacbf095e1c715417de555c123d))
* 中文readme标题显示问题 ([ce2f71a](https://github.com/nicokim/indextts2-inference/commit/ce2f71aae5b568c17428903aa3f8890c10fe4d26))
* 修复样本音频太长报错的问题，对音频进行裁切。 ([2cfc76a](https://github.com/nicokim/indextts2-inference/commit/2cfc76ad9c49b4376e3d10a5bfeafbad9e5b5cd3))
* 添加英语缩写处理 ([414f2a4](https://github.com/nicokim/indextts2-inference/commit/414f2a4052b49248a4ff99380bb84040d0c6e22a))
* 添加英语缩写处理 ([bb4d76a](https://github.com/nicokim/indextts2-inference/commit/bb4d76aa2a1b5e3efb00ccc6ee7fc45e453f2701))
* 避免在 MinGW-w64 环境 jit compile cuda ext ([92bb2eb](https://github.com/nicokim/indextts2-inference/commit/92bb2eb0c0a5242943fb86950ce68c73576053a4))


### Documentation

* Add a stronger warning about unsupported installation methods ([6c76807](https://github.com/nicokim/indextts2-inference/commit/6c768073e9519fa253090d539779e2d31be03cf5))
* Add Alibaba's high-bandwidth PyPI mirror for China ([30848ef](https://github.com/nicokim/indextts2-inference/commit/30848efd45f91d55b7f79f7f4a9388c74077afd7))
* Add FP16 usage advice for faster inference ([d777b8a](https://github.com/nicokim/indextts2-inference/commit/d777b8a0290cced0064eb5f503a25b65b148c1bc))
* Add quick uv installation technique ([429c06c](https://github.com/nicokim/indextts2-inference/commit/429c06c787115b4036330181e392785aa9b971f9))
* Add usage note regarding random sampling ([c3d7ab4](https://github.com/nicokim/indextts2-inference/commit/c3d7ab4adce98e8f46edb1a31374f2f64d72f51c))
* Clarify that UV handles Python and the environment creation ([cc9c6b6](https://github.com/nicokim/indextts2-inference/commit/cc9c6b6cfe49c6d9e6f84247b785ac08a2c808f5))
* Document the DeepSpeed performance effects ([85ba55a](https://github.com/nicokim/indextts2-inference/commit/85ba55a1d3ac92006f898115600f2cfef4e34251))
* Document the new `emo_alpha` feature for text-to-emotion mode ([3b5b6bc](https://github.com/nicokim/indextts2-inference/commit/3b5b6bca85ad01955bf773686647c3983e737dd9))
* Install HuggingFace CLI with high-speed download feature ([5471d82](https://github.com/nicokim/indextts2-inference/commit/5471d8256fc7867772030b1c0c75fe633f49ea76))
* Remove redundant "python" command instruction ([242604d](https://github.com/nicokim/indextts2-inference/commit/242604d27e63fba64df00985eb365013e7019bbe))
* Remove redundant "python" command instruction ([3236fa4](https://github.com/nicokim/indextts2-inference/commit/3236fa496a1d3ed587feb2b24fc85877aea51554))

## [1.0.0](https://github.com/nicokim/indextts2-inference/compare/index-tts-inference-v0.1.0...index-tts-inference-v1.0.0) (2026-02-26)


### ⚠ BREAKING CHANGES

* de-vendor transformers, support transformers 5.x ([#2](https://github.com/nicokim/indextts2-inference/issues/2))
* **webui:** Easier DeepSpeed launch argument

### Features

* **accel:** add batch support ([44b5ffc](https://github.com/nicokim/indextts2-inference/commit/44b5ffcb5dda05217e08521d58a919a44c6dac2b))
* achieve inference acceleration for the gpt2 stage ([c1ef414](https://github.com/nicokim/indextts2-inference/commit/c1ef4148af34dcb1943ae1fe4bded0f7602f68bb))
* achieve inference acceleration for the gpt2 stage (3.79×) ([1d5d079](https://github.com/nicokim/indextts2-inference/commit/1d5d079aaa290a7179967e80b7e25f5aa002cd8f))
* achieve inference acceleration for the s2mel stage (1.61×) ([e42480c](https://github.com/nicokim/indextts2-inference/commit/e42480ced8af8bf8ef596b2420faa5c86fa64ed2))
* add batch support ([3c36027](https://github.com/nicokim/indextts2-inference/commit/3c360273da9738158ee72bf6d47710033f835049))
* Add reusable Emotion Vector normalization helper ([8aa8064](https://github.com/nicokim/indextts2-inference/commit/8aa8064a53c5b5b53ff20f6de94aaadc18a4cd9d))
* **cli:** Support XPU ([#322](https://github.com/nicokim/indextts2-inference/issues/322)) ([e83df4e](https://github.com/nicokim/indextts2-inference/commit/e83df4e4270e6b1f5f08b341907689c078a382c4))
* de-vendor transformers, support transformers 5.x ([#2](https://github.com/nicokim/indextts2-inference/issues/2)) ([ad522fe](https://github.com/nicokim/indextts2-inference/commit/ad522fe80dad9aefd87ce7c233d02b17f6e11733))
* DeepSpeed is now an optional dependency which can be disabled ([936e6ac](https://github.com/nicokim/indextts2-inference/commit/936e6ac4dd4558b29c431e709c57a04cbd9c78bc))
* Extend GPU Check utility to support more GPUs ([39a035d](https://github.com/nicokim/indextts2-inference/commit/39a035d1066dee3baeb45a5580555ea2013591f0))
* **front.py:** add regex pattern for technical terms ([82a5b90](https://github.com/nicokim/indextts2-inference/commit/82a5b9004a90bb6a9656cfb95e3aaa05d1117a0a))
* gumbel_softmax_sampler ([a3b884f](https://github.com/nicokim/indextts2-inference/commit/a3b884ff6ff401923e6242fb87630be0a6d0f221))
* **i18n:** Add missing UI translation strings ([5f0b0a9](https://github.com/nicokim/indextts2-inference/commit/5f0b0a9f9c57c05440d6d3992e90792d70023ae5))
* Implement `emo_alpha` scaling of emotion vectors and emotion text ([9668064](https://github.com/nicokim/indextts2-inference/commit/9668064377fcba7d3e8643c8680d5f8d37ddb2a9))
* **indextts:** add glossary support for custom term pronunciations ([6deed97](https://github.com/nicokim/indextts2-inference/commit/6deed97efe636bf68825ebb0ec82ec994fd1d42f))
* optimize s2mel stage ([31e7e85](https://github.com/nicokim/indextts2-inference/commit/31e7e855e21779c5ad0cd6b7b0ae4329c0c91ec2))
* **sampler:** enhance with greedy sampling mode ([42a7339](https://github.com/nicokim/indextts2-inference/commit/42a73394e9f9049699bc8e5210a705249e248b3f))
* Warn if input text contains UNK tokens ([34be9bf](https://github.com/nicokim/indextts2-inference/commit/34be9bfb146ed4cf41808f002fc47332ddddaebd))
* **webui.py:** add glossary term validation and error handling ([a7099c4](https://github.com/nicokim/indextts2-inference/commit/a7099c4a6e13a59b2b4e9c8e90e06b61595412f0))
* **webui:** Easier DeepSpeed launch argument ([f0badb1](https://github.com/nicokim/indextts2-inference/commit/f0badb13af7f84f92caa4d7d65b56ef4e3456337))
* **webui:** Implement emotion weighting for vectors and text modes ([d899770](https://github.com/nicokim/indextts2-inference/commit/d8997703137434e2d71329721e52e88eff534572))
* **webui:** Implement speech synthesis progress bar ([555e146](https://github.com/nicokim/indextts2-inference/commit/555e146fb4066d465225a850b21243e5b7eccd0e))
* 归一化参数到推荐的范围，改善用户体验 ([48a71af](https://github.com/nicokim/indextts2-inference/commit/48a71aff6da497c3d14c9a0bd2be9d627c8c45d3))
* 归一化参数到推荐的范围，改善用户体验 ([af2b06e](https://github.com/nicokim/indextts2-inference/commit/af2b06e061937d291e276a0aa1bd4a2254a06078))
* 裁剪过长的输入音频至15s,减少爆内存和显存 ([009428b](https://github.com/nicokim/indextts2-inference/commit/009428b62dc60b9d7bfdddd1641d51df9c2afa80))
* 裁剪过长的输入音频至15s,减少爆内存和显存 ([0828dcb](https://github.com/nicokim/indextts2-inference/commit/0828dcb098247760ba17ba5ad8a5b6b5cb460f95))


### Bug Fixes

* add force_rebuild flag for fused alias_free_activation and update installation instructions ([59c05c0](https://github.com/nicokim/indextts2-inference/commit/59c05c0765b61f4c8fe5d66ff7fe3c6a960e09e4))
* Add support for melancholic emotion in text-to-emotion vectors ([a6a955d](https://github.com/nicokim/indextts2-inference/commit/a6a955d2aa18d6400b2f4cdaad35e792d5bb231b))
* **cli:** More robust device priority checks ([6113567](https://github.com/nicokim/indextts2-inference/commit/6113567e94a885193f451d3d81665a609e9efbe5))
* Don't load DeepSpeed if use_deepspeed is False ([05a8ae4](https://github.com/nicokim/indextts2-inference/commit/05a8ae45e5da8ab7bdb035069181c11dfa3ac2bf))
* Empty generator -&gt; IndexError problem on non-streaming infer() ([db5b39b](https://github.com/nicokim/indextts2-inference/commit/db5b39bb6ad903c219b2dd33d60b0f0bdaede664))
* Empty generator -&gt; IndexError problem on non-streaming infer() ([750d9d9](https://github.com/nicokim/indextts2-inference/commit/750d9d9d15c90cfbf2d1f386a481e0087a834a61))
* Fast and robust text-to-emotion algorithm ([58ad225](https://github.com/nicokim/indextts2-inference/commit/58ad225fb41c99ebf873f3b1694391f8164ac0d3))
* Fix character encoding in examples ([55b7d32](https://github.com/nicokim/indextts2-inference/commit/55b7d321495b530c5467ecdf0e06fb996a011a71))
* Fix internal text-to-emotion vector labels ([feba501](https://github.com/nicokim/indextts2-inference/commit/feba5010137460ec8decaba2603da7819cb963e6))
* **front.py:** load full term glossary entries from yaml file ([1460adb](https://github.com/nicokim/indextts2-inference/commit/1460adbdc54d13af39b155e2760d0045ac23a825))
* handle multiple sentence placeholders in de_tokenized_by_CJK_char ([267e344](https://github.com/nicokim/indextts2-inference/commit/267e344a0956c09f4c2ff9e16d1c6bda7d67c11a))
* Improve .gitignore and re-add config file ([5ffb84b](https://github.com/nicokim/indextts2-inference/commit/5ffb84b427d2d7e8fa597cc4f5e31c2b30cfd492))
* **infer_v2:** Correct the import path of BigVGAN's custom cuda kernel ([e409c4a](https://github.com/nicokim/indextts2-inference/commit/e409c4a19b911f203de69f043987e9f37d78b0c8))
* Suppress pandas PyArrow future dependency warning ([d5cdb5e](https://github.com/nicokim/indextts2-inference/commit/d5cdb5eb3cfb34cbb901e4da4a0302c07e6a5e06))
* Update pandas to fix Gradio errors ([3e64c4a](https://github.com/nicokim/indextts2-inference/commit/3e64c4ac11c12d119567fa3a1b47da577170fbc8))
* update PINYIN_TONE_PATTERN and NormalizerZh ([7d943b3](https://github.com/nicokim/indextts2-inference/commit/7d943b362dbc950edc2dfc1fb7726dbfd5d39b81))
* Use WeTextProcessing on Linux, and wetext on other platforms ([dcdb061](https://github.com/nicokim/indextts2-inference/commit/dcdb0614bf7fc29fe560969324478c29928b1695))
* **webui.py:** replace return statements with warnings and update markdown table ([fa7962f](https://github.com/nicokim/indextts2-inference/commit/fa7962f1f2decd7a430d2d62544d2e3a3e596d06))
* **webui.py:** strip tailing whitespace for glossary terms ([6b6606a](https://github.com/nicokim/indextts2-inference/commit/6b6606a2f4d1974ea8ae1413e609b7a17af882fc))
* **webui:** Add support for Gradio 5.45.0 and higher ([ef09710](https://github.com/nicokim/indextts2-inference/commit/ef097101b7ab70120250c9c3f6394174e656590e))
* **webui:** Experimental checkbox bugfixes and add visual warning label ([ec368de](https://github.com/nicokim/indextts2-inference/commit/ec368de9329c7beffa3df369005c7d25e7684980))
* **webui:** Fix unintentional empty spacing between control groups ([f041d8e](https://github.com/nicokim/indextts2-inference/commit/f041d8eb64e1af1f7a5def01247d564b1a283a5b))
* **webui:** Make the Advanced Settings visible by default again ([e185fa1](https://github.com/nicokim/indextts2-inference/commit/e185fa1ce748de6759f772f3852024c5443724f3))
* **webui:** Make the Emotion Control Weight slider visible again ([c5f9a31](https://github.com/nicokim/indextts2-inference/commit/c5f9a311275447707791c2758a2267dd137142f2))
* **webui:** New default emo_alpha recommendation instead of scaling ([1520d06](https://github.com/nicokim/indextts2-inference/commit/1520d0689baa4bacbf095e1c715417de555c123d))
* 中文readme标题显示问题 ([ce2f71a](https://github.com/nicokim/indextts2-inference/commit/ce2f71aae5b568c17428903aa3f8890c10fe4d26))
* 修复样本音频太长报错的问题，对音频进行裁切。 ([2cfc76a](https://github.com/nicokim/indextts2-inference/commit/2cfc76ad9c49b4376e3d10a5bfeafbad9e5b5cd3))
* 添加英语缩写处理 ([414f2a4](https://github.com/nicokim/indextts2-inference/commit/414f2a4052b49248a4ff99380bb84040d0c6e22a))
* 添加英语缩写处理 ([bb4d76a](https://github.com/nicokim/indextts2-inference/commit/bb4d76aa2a1b5e3efb00ccc6ee7fc45e453f2701))
* 避免在 MinGW-w64 环境 jit compile cuda ext ([92bb2eb](https://github.com/nicokim/indextts2-inference/commit/92bb2eb0c0a5242943fb86950ce68c73576053a4))


### Documentation

* Add a stronger warning about unsupported installation methods ([6c76807](https://github.com/nicokim/indextts2-inference/commit/6c768073e9519fa253090d539779e2d31be03cf5))
* Add Alibaba's high-bandwidth PyPI mirror for China ([30848ef](https://github.com/nicokim/indextts2-inference/commit/30848efd45f91d55b7f79f7f4a9388c74077afd7))
* Add FP16 usage advice for faster inference ([d777b8a](https://github.com/nicokim/indextts2-inference/commit/d777b8a0290cced0064eb5f503a25b65b148c1bc))
* Add quick uv installation technique ([429c06c](https://github.com/nicokim/indextts2-inference/commit/429c06c787115b4036330181e392785aa9b971f9))
* Add usage note regarding random sampling ([c3d7ab4](https://github.com/nicokim/indextts2-inference/commit/c3d7ab4adce98e8f46edb1a31374f2f64d72f51c))
* Clarify that UV handles Python and the environment creation ([cc9c6b6](https://github.com/nicokim/indextts2-inference/commit/cc9c6b6cfe49c6d9e6f84247b785ac08a2c808f5))
* Document the DeepSpeed performance effects ([85ba55a](https://github.com/nicokim/indextts2-inference/commit/85ba55a1d3ac92006f898115600f2cfef4e34251))
* Document the new `emo_alpha` feature for text-to-emotion mode ([3b5b6bc](https://github.com/nicokim/indextts2-inference/commit/3b5b6bca85ad01955bf773686647c3983e737dd9))
* Install HuggingFace CLI with high-speed download feature ([5471d82](https://github.com/nicokim/indextts2-inference/commit/5471d8256fc7867772030b1c0c75fe633f49ea76))
* Remove redundant "python" command instruction ([242604d](https://github.com/nicokim/indextts2-inference/commit/242604d27e63fba64df00985eb365013e7019bbe))
* Remove redundant "python" command instruction ([3236fa4](https://github.com/nicokim/indextts2-inference/commit/3236fa496a1d3ed587feb2b24fc85877aea51554))

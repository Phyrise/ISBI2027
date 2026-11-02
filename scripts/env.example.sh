# Copy, edit and source before running the scripts:  source scripts/env.sh
export ISBI_DATA=/path/to/preprocessed/duke_train_dev     # output of: python -m synthesis.data.preprocess --splits train dev
export ISBI_TEST_DATA=/path/to/preprocessed/duke_test91   # output of: python -m synthesis.data.preprocess --splits test
export ISBI_RUNS=/path/to/runs                            # one folder per arm / expert (checkpoints, logs)
export ISBI_WEIGHTS=/path/to/weights                      # nnFoundationCNN/checkpoint_final.pth, resnet_50_23dataset.pth,
                                                          # authors_SAM2.1_Small.pt, authors_VGG16.pt
# export ISBI_INITS=/path/to/inits                        # optional: generator_init.pth, generator_init_swin_r11.pth,
                                                          # discriminator_init_r11.pth (else redrawn from their seeds)

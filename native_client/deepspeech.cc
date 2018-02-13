#ifdef DS_NATIVE_MODEL
#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "native_client/deepspeech_model_core.h" // generated
#endif

#include <iostream>

#include "deepspeech.h"
#include "deepspeech_utils.h"
#include "alphabet.h"
#include "beam_search.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/memmapped_file_system.h"

#include "c_speech_features.h"

//TODO: infer batch size from model/use dynamic batch size
const int BATCH_SIZE = 1;

//TODO: use dynamic sample rate
const int SAMPLE_RATE = 16000;

//TODO: infer n_steps from model
const int N_STEPS_PER_BATCH = 16;

const float AUDIO_WIN_LEN = 0.025f;
const float AUDIO_WIN_STEP = 0.01f;
const int AUDIO_WIN_LEN_SAMPLES = (int)(AUDIO_WIN_LEN * SAMPLE_RATE);
const int AUDIO_WIN_STEP_SAMPLES = (int)(AUDIO_WIN_STEP * SAMPLE_RATE);

const int MFCC_FEATURES = 26;
const int MFCC_CONTEXT = 9;
const int MFCC_WIN_LEN = 2 * MFCC_CONTEXT + 1;
const int MFCC_FEATS_PER_TIMESTEP = MFCC_FEATURES * MFCC_WIN_LEN;

const float COEFF = 0.97f;
const int N_FFT = 512;
const int N_FILTERS = 26;
const int LOWFREQ = 0;
const int CEP_LIFTER = 22;

using namespace tensorflow;
using tensorflow::ctc::CTCBeamSearchDecoder;
using tensorflow::ctc::CTCDecoder;

namespace DeepSpeech {

struct StreamingState {
  float* accumulated_logits;
  unsigned int capacity_timesteps;
  unsigned int current_timestep;
  short audio_buffer[AUDIO_WIN_LEN_SAMPLES];
  unsigned int audio_buffer_len;
  float mfcc_buffer[MFCC_FEATS_PER_TIMESTEP];
  unsigned int mfcc_buffer_len;
  float batch_buffer[N_STEPS_PER_BATCH*MFCC_FEATS_PER_TIMESTEP];
  unsigned int batch_buffer_len;
  bool skip_next_mfcc;
};

struct Private {
  MemmappedEnv* mmap_env;
  Session* session;
  GraphDef graph_def;
  int ncep;
  int ncontext;
  Alphabet* alphabet;
  KenLMBeamScorer* scorer;
  int beam_width;
  bool run_aot;

  /* This is the actual implementation of the streaming inference API, with the
     Model class just forwarding the calls to this class.

     The streaming process uses three buffers that are fed eagerly as audio data
     is fed in. The buffers only hold the minimum amount of data needed to do a
     step in the acoustic model. The three buffers which live in StreamingContext
     are:

     - audio_buffer, used to buffer audio samples until there's enough data to
       compute input features for a single window.

     - mfcc_buffer, used to buffer input features until there's enough data for
       a single timestep. Remember there's overlap in the features, each timestep
       contains N_CONTEXT past feature frames, the current feature frame, and
       N_CONTEXT future feature frames, for a total of MFCC_WIN_LEN feature frames
       per timestep.

     - batch_buffer, used to buffer timesteps until there's enough data to compute
       a batch of N_STEPS_PER_BATCH.

     Data flows through all three buffers as audio samples are fed via the public
     API. When audio_buffer is full, features are computed from it and pushed to
     mfcc_buffer. When mfcc_buffer is full, the timestep is copied to batch_buffer.
     When batch_buffer is full, we do a single step through the acoustic model
     and accumulate results in StreamingState::accumulated_logits.

     When fininshStream() is called, we decode the accumulated logits and return
     the corresponding transcription.
  */
  StreamingState* setupStream(unsigned int prealloc_frames, unsigned int sample_rate);
  void feedAudioContent(StreamingState* ctx, const short* buffer, unsigned int buffer_size);
  char* finishStream(StreamingState* ctx);

private:
  /**
   * @brief Perform decoding of the logits, using basic CTC decoder or
   *        CTC decoder with KenLM enabled
   *
   * @param n_frames       Number of timesteps to deal with
   * @param logits         Matrix of logits, of dimensions:
   *                       [n_frames][batch_size][num_classes]
   *
   * @return String representing the decoded text.
   *
   * @note This method will free @p logits.
   */
  char* decode(int n_frames, float* logits);

  /**
   * @brief Do a single inference step in the acoustic model, with:
   *          input=mfcc
   *          input_lengths=[n_frames]
   *
   * @param mfcc batch input data
   * @param n_frames number of timesteps in the data
   *
   * @param[out] output_logits Should be large enough to fit
   *                           aNFrames * alphabet_size floats.
   */
  void infer(float* mfcc, int n_frames, float* output_logits);

  void processAudioWindow(StreamingState* ctx, short* buf, unsigned int len);
  void processMfccWindow(StreamingState* ctx, float* buf, unsigned int len);
  void addZeroMfccWindow(StreamingState* ctx);
  void processBatch(StreamingState* ctx, float* buf, unsigned int len);
};

StreamingState*
Private::setupStream(unsigned int prealloc_frames,
                     unsigned int /*sample_rate*/)
{
  Status status = session->Run({}, {}, {"initialize_state"}, nullptr);

  if (!status.ok()) {
    std::cerr << "Error running session: " << status << "\n";
    return nullptr;
  }

  StreamingState* ctx = new StreamingState;
  const size_t num_classes = alphabet->GetSize() + 1; // +1 for blank

  float* logits = (float*)malloc(prealloc_frames * BATCH_SIZE * num_classes * sizeof(float));

  ctx->accumulated_logits = logits;
  ctx->capacity_timesteps = prealloc_frames;
  ctx->current_timestep = 0;

  memset(ctx->audio_buffer, 0, sizeof(ctx->audio_buffer));
  ctx->audio_buffer_len = 0;

  memset(ctx->mfcc_buffer, 0, sizeof(ctx->mfcc_buffer));
  ctx->mfcc_buffer_len = MFCC_FEATURES*MFCC_CONTEXT; // initial zero entries

  memset(ctx->batch_buffer, 0, sizeof(ctx->batch_buffer));
  ctx->batch_buffer_len = 0;

  ctx->skip_next_mfcc = false;

  return ctx;
}

void
Private::feedAudioContent(StreamingState* ctx,
                          const short* buffer,
                          unsigned int buffer_size)
{
  while (buffer_size > 0) {
    while (buffer_size > 0 && ctx->audio_buffer_len < AUDIO_WIN_LEN_SAMPLES) {
      ctx->audio_buffer[ctx->audio_buffer_len] = *buffer;
      ++ctx->audio_buffer_len;
      ++buffer;
      --buffer_size;
    }

    if (ctx->audio_buffer_len == AUDIO_WIN_LEN_SAMPLES) {
      processAudioWindow(ctx, ctx->audio_buffer, ctx->audio_buffer_len);
      // shift data by one step of 10ms
      memmove(ctx->audio_buffer, ctx->audio_buffer + AUDIO_WIN_STEP_SAMPLES, (AUDIO_WIN_LEN_SAMPLES - AUDIO_WIN_STEP_SAMPLES) * sizeof(short));
      ctx->audio_buffer_len -= AUDIO_WIN_STEP_SAMPLES;
    }
  }
}

char*
Private::finishStream(StreamingState* ctx)
{
  // flush audio buffer
  processAudioWindow(ctx, ctx->audio_buffer, ctx->audio_buffer_len);

  // add empty mfcc vectors at end of sample
  for (int i = 0; i < MFCC_CONTEXT; ++i) {
    addZeroMfccWindow(ctx);
  }

  // process final batch
  if (ctx->batch_buffer_len > 0) {
    processBatch(ctx, ctx->batch_buffer, ctx->batch_buffer_len/MFCC_FEATS_PER_TIMESTEP);
  }

  char* str = decode(ctx->current_timestep, ctx->accumulated_logits);
  delete ctx;
  return str;
}

void
Private::processAudioWindow(StreamingState* ctx, short* buf, unsigned int len)
{
  ctx->skip_next_mfcc = !ctx->skip_next_mfcc;
  if (!ctx->skip_next_mfcc) { // was true
    return;
  }

  // Compute MFCC features
  float* mfcc;
  int n_frames = csf_mfcc(buf, len, SAMPLE_RATE,
                          AUDIO_WIN_LEN, AUDIO_WIN_STEP, MFCC_FEATURES, N_FILTERS, N_FFT,
                          LOWFREQ, SAMPLE_RATE/2, COEFF, CEP_LIFTER, 1, nullptr,
                          &mfcc);
  assert(n_frames == 1);

  float* buffer = mfcc;
  unsigned int bufferSize = n_frames * MFCC_FEATURES;
  while (bufferSize > 0) {
    while (bufferSize > 0 && ctx->mfcc_buffer_len < MFCC_FEATS_PER_TIMESTEP) {
      ctx->mfcc_buffer[ctx->mfcc_buffer_len] = *buffer;
      ++ctx->mfcc_buffer_len;
      ++buffer;
      --bufferSize;
    }

    if (ctx->mfcc_buffer_len == MFCC_FEATS_PER_TIMESTEP) {
      processMfccWindow(ctx, ctx->mfcc_buffer, ctx->mfcc_buffer_len);
      // shift data by one step of one mfcc feature vector
      memmove(ctx->mfcc_buffer, ctx->mfcc_buffer + MFCC_FEATURES, (ctx->mfcc_buffer_len - MFCC_FEATURES) * sizeof(float));
      ctx->mfcc_buffer_len -= MFCC_FEATURES;
    }
  }

  free(mfcc);
}

void
Private::processMfccWindow(StreamingState* ctx, float* buf, unsigned int len)
{
  while (len > 0) {
    while (len > 0 && ctx->batch_buffer_len < N_STEPS_PER_BATCH*MFCC_FEATS_PER_TIMESTEP) {
      ctx->batch_buffer[ctx->batch_buffer_len] = *buf;
      ++ctx->batch_buffer_len;
      ++buf;
      --len;
    }

    if (ctx->batch_buffer_len == N_STEPS_PER_BATCH*MFCC_FEATS_PER_TIMESTEP) {
      processBatch(ctx, ctx->batch_buffer, N_STEPS_PER_BATCH);
      memset(ctx->batch_buffer, 0, N_STEPS_PER_BATCH*MFCC_FEATS_PER_TIMESTEP*sizeof(float));
      ctx->batch_buffer_len = 0;
    }
  }
}

void
Private::addZeroMfccWindow(StreamingState* ctx)
{
  unsigned int bufferSize = MFCC_FEATURES;
  while (bufferSize > 0) {
    while (bufferSize > 0 && ctx->mfcc_buffer_len < MFCC_FEATS_PER_TIMESTEP) {
      ctx->mfcc_buffer[ctx->mfcc_buffer_len] = 0.f;
      ++ctx->mfcc_buffer_len;
      --bufferSize;
    }

    if (ctx->mfcc_buffer_len == MFCC_FEATS_PER_TIMESTEP) {
      processMfccWindow(ctx, ctx->mfcc_buffer, ctx->mfcc_buffer_len);
      // shift data by one step of one mfcc feature vector
      memmove(ctx->mfcc_buffer, ctx->mfcc_buffer + MFCC_FEATURES, (ctx->mfcc_buffer_len - MFCC_FEATURES) * sizeof(float));
      ctx->mfcc_buffer_len -= MFCC_FEATURES;
    }
  }
}

void
Private::processBatch(StreamingState* ctx, float* buf, unsigned int n_steps)
{
  const size_t num_classes = alphabet->GetSize() + 1; // +1 for blank

  if (ctx->current_timestep >= ctx->capacity_timesteps-n_steps) {
    ctx->capacity_timesteps = ctx->capacity_timesteps * 2;
    ctx->accumulated_logits = (float*)realloc(ctx->accumulated_logits, ctx->capacity_timesteps * BATCH_SIZE * num_classes * sizeof(float));
  }

  infer(buf, n_steps, &ctx->accumulated_logits[ctx->current_timestep * BATCH_SIZE * num_classes]);
  ctx->current_timestep += n_steps;
}

void
Private::infer(float* aMfcc, int n_frames, float* logits_output)
{
  const size_t num_classes = alphabet->GetSize() + 1; // +1 for blank

  if (run_aot) {
#ifdef DS_NATIVE_MODEL
    Eigen::ThreadPool tp(2);  // Size the thread pool as appropriate.
    Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());

    nativeModel nm(nativeModel::AllocMode::RESULTS_PROFILES_AND_TEMPS_ONLY);
    nm.set_thread_pool(&device);

    for (int ot = 0; ot < n_frames; ot += DS_MODEL_TIMESTEPS) {
      nm.set_arg0_data(&(aMfcc[ot * MFCC_FEATS_PER_TIMESTEP]));
      nm.Run();

      // The CTCDecoder works with log-probs.
      for (int t = 0; t < DS_MODEL_TIMESTEPS, (ot + t) < n_frames; ++t) {
        for (int b = 0; b < BATCH_SIZE; ++b) {
          for (int c = 0; c < num_classes; ++c) {
            logits_output[((ot + t) * BATCH_SIZE * num_classes) + (b * num_classes) + c] = nm.result0(t, b, c);
          }
        }
      }
    }
#else
    std::cerr << "No support for native model built-in." << std::endl;
    return;
#endif // DS_NATIVE_MODEL
  } else {
    Tensor input(DT_FLOAT, TensorShape({BATCH_SIZE, N_STEPS_PER_BATCH, MFCC_FEATS_PER_TIMESTEP}));

    auto input_mapped = input.tensor<float, 3>();
    int idx = 0;
    for (int i = 0; i < n_frames; i++) {
      for (int j = 0; j < MFCC_FEATS_PER_TIMESTEP; j++, idx++) {
        input_mapped(0, i, j) = aMfcc[idx];
      }
    }

    Tensor input_lengths(DT_INT32, TensorShape({1}));
    input_lengths.scalar<int>()() = n_frames;

    std::vector<Tensor> outputs;
    Status status = session->Run(
      {{"input_node", input}, {"input_lengths", input_lengths}},
      {"logits"}, {}, &outputs);

    if (!status.ok()) {
      std::cerr << "Error running session: " << status << "\n";
      return;
    }

    auto logits_mapped = outputs[0].tensor<float, 3>();
    // The CTCDecoder works with log-probs.
    for (int t = 0; t < n_frames; ++t) {
      for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int c = 0; c < num_classes; ++c) {
          logits_output[(t * BATCH_SIZE * num_classes) + (b * num_classes) + c] = logits_mapped(t, b, c);
        }
      }
    }
  }
}

char*
Private::decode(int n_frames, float* logits)
{
  const int top_paths = 1;
  const size_t num_classes = alphabet->GetSize() + 1; // +1 for blank

  // Raw data containers (arrays of floats, ints, etc.).
  int sequence_lengths[BATCH_SIZE] = {n_frames};

  // Convert data containers to the format accepted by the decoder, simply
  // mapping the memory from the container to an Eigen::ArrayXi,::MatrixXf,
  // using Eigen::Map.
  Eigen::Map<const Eigen::ArrayXi> seq_len(&sequence_lengths[0], BATCH_SIZE);
  std::vector<Eigen::Map<const Eigen::MatrixXf>> inputs;
  inputs.reserve(n_frames);
  for (int t = 0; t < n_frames; ++t) {
    inputs.emplace_back(&logits[t * BATCH_SIZE * num_classes], BATCH_SIZE, num_classes);
  }

  // Prepare containers for output and scores.
  // CTCDecoder::Output is std::vector<std::vector<int>>
  std::vector<CTCDecoder::Output> decoder_outputs(top_paths);
  for (CTCDecoder::Output& output : decoder_outputs) {
    output.resize(BATCH_SIZE);
  }
  float score[BATCH_SIZE][top_paths] = {{0.0}};
  Eigen::Map<Eigen::MatrixXf> scores(&score[0][0], BATCH_SIZE, top_paths);

  if (scorer == nullptr) {
    CTCBeamSearchDecoder<>::DefaultBeamScorer default_scorer;
    CTCBeamSearchDecoder<> decoder(num_classes,
                                   beam_width,
                                   &default_scorer,
                                   BATCH_SIZE);
    decoder.Decode(seq_len, inputs, &decoder_outputs, &scores).ok();
  } else {
    CTCBeamSearchDecoder<KenLMBeamState> decoder(num_classes,
                                                 beam_width,
                                                 scorer,
                                                 BATCH_SIZE);
    decoder.Decode(seq_len, inputs, &decoder_outputs, &scores).ok();
  }

  free(logits);

  // Output is an array of shape (1, n_results, result_length).
  // In this case, n_results is also equal to 1.
  size_t output_length = decoder_outputs[0][0].size() + 1;

  std::stringstream output;
  for (int i = 0; i < output_length - 1; i++) {
    int64 character = decoder_outputs[0][0][i];
    output << alphabet->StringFromLabel(character);
  }

  return strdup(output.str().c_str());
}

DEEPSPEECH_EXPORT
Model::Model(const char* aModelPath, int aNCep, int aNContext,
             const char* aAlphabetConfigPath, int aBeamWidth)
{
  mPriv             = new Private;
  mPriv->mmap_env   = new MemmappedEnv(Env::Default());
  mPriv->session    = nullptr;
  mPriv->scorer     = nullptr;
  mPriv->ncep       = aNCep;
  mPriv->ncontext   = aNContext;
  mPriv->alphabet   = new Alphabet(aAlphabetConfigPath);
  mPriv->beam_width = aBeamWidth;
  mPriv->run_aot    = false;

  if (!aModelPath || strlen(aModelPath) < 1) {
    std::cerr << "No model specified, will rely on built-in model." << std::endl;
    mPriv->run_aot = true;
    return;
  }

  Status status;
  SessionOptions options;

  bool is_mmap = std::string(aModelPath).find(".pbmm") != std::string::npos;
  if (!is_mmap) {
    std::cerr << "Warning: reading entire model file into memory. Transform model file into an mmapped graph to reduce heap usage." << std::endl;
  } else {
    status = mPriv->mmap_env->InitializeFromFile(aModelPath);
    if (!status.ok()) {
      std::cerr << status << std::endl;
      return;
    }

    options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(::OptimizerOptions::L0);
    options.env = mPriv->mmap_env;
  }

  status = NewSession(options, &mPriv->session);
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return;
  }

  if (is_mmap) {
    status = ReadBinaryProto(mPriv->mmap_env,
                             MemmappedFileSystem::kMemmappedPackageDefaultGraphDef,
                             &mPriv->graph_def);
  } else {
    status = ReadBinaryProto(Env::Default(), aModelPath, &mPriv->graph_def);
  }
  if (!status.ok()) {
    (void)mPriv->session->Close();
    mPriv->session = nullptr;
    std::cerr << status << std::endl;
    return;
  }

  status = mPriv->session->Create(mPriv->graph_def);
  if (!status.ok()) {
    (void)mPriv->session->Close();
    mPriv->session = nullptr;
    std::cerr << status << std::endl;
    return;
  }

  for (int i = 0; i < mPriv->graph_def.node_size(); ++i) {
    NodeDef node = mPriv->graph_def.node(i);
    if (node.name() == "logits_shape") {
      Tensor logits_shape = Tensor(DT_INT32, TensorShape({3}));
      if (!logits_shape.FromProto(node.attr().at("value").tensor())) {
        break;
      }

      int final_dim_size = logits_shape.vec<int>()(2) - 1;
      if (final_dim_size != mPriv->alphabet->GetSize()) {
        std::cerr << "Error: Alphabet size does not match loaded model: alphabet "
                  << "has size " << mPriv->alphabet->GetSize()
                  << ", but model has " << final_dim_size
                  << " classes in its output. Make sure you're passing an alphabet "
                  << "file with the same size as the one used for training."
                  << std::endl;
        (void)mPriv->session->Close();
        mPriv->session = nullptr;
        return;
      }
      break;
    }
  }
}

DEEPSPEECH_EXPORT
Model::~Model()
{
  if (mPriv->session) {
    (void)mPriv->session->Close();
  }

  delete mPriv->mmap_env;
  delete mPriv->alphabet;
  delete mPriv->scorer;

  delete mPriv;
}

DEEPSPEECH_EXPORT
void
Model::enableDecoderWithLM(const char* aAlphabetConfigPath, const char* aLMPath,
                           const char* aTriePath, float aLMWeight,
                           float aWordCountWeight, float aValidWordCountWeight)
{
  mPriv->scorer = new KenLMBeamScorer(aLMPath, aTriePath, aAlphabetConfigPath,
                                      aLMWeight, aWordCountWeight, aValidWordCountWeight);
}

DEEPSPEECH_EXPORT
void
Model::getInputVector(const short* aBuffer, unsigned int aBufferSize,
                      int aSampleRate, float** aMfcc, int* aNFrames,
                      int* aFrameLen)
{
  return audioToInputVector(aBuffer, aBufferSize, aSampleRate, mPriv->ncep,
                            mPriv->ncontext, aMfcc, aNFrames, aFrameLen);
}

DEEPSPEECH_EXPORT
char*
Model::stt(const short* aBuffer,
           unsigned int aBufferSize,
           int aSampleRate)
{
  StreamingState* ctx = setupStream();
  feedAudioContent(ctx, aBuffer, aBufferSize);
  return finishStream(ctx);
}

DEEPSPEECH_EXPORT
StreamingState*
Model::setupStream(unsigned int aPreAllocFrames,
                   unsigned int aSampleRate)
{
  return mPriv->setupStream(aPreAllocFrames, aSampleRate);
}

DEEPSPEECH_EXPORT
void
Model::feedAudioContent(StreamingState* ctx,
                        const short* aBuffer,
                        unsigned int aBufferSize)
{
  mPriv->feedAudioContent(ctx, aBuffer, aBufferSize);
}

DEEPSPEECH_EXPORT
char*
Model::finishStream(StreamingState* ctx)
{
  return mPriv->finishStream(ctx);
}

}

# Project Write-Up

## Explaining Custom Layers

Model Optimizer searches for each layer in input model in the list of known layers, before building it's Intermediate Representation.
If that layer is not found it is classified as Custom Layer.
To add support involves registering custom layer as extension to the Model Optimizer. When registered it generated valid & optimized IR.
Though the procedure is different for different frameworks (See [here](https://docs.openvinotoolkit.org/2020.2/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html)).

There is also second option for frameworks too-
* For caffe, register layer as custom and use system Caffe to calculate the output shape of each such layer
* For Tensorflow, replace it's unsupported subgraph with a different subgraph. Or third option to offload computation to Tesorflow itself.
* For MXNet models too, another analogous subgraph should replace the unsupported one.

Some of the potential reasons for handling custom layers are-
* Some layers not supported but required for inference.
* Allows to deploy models in which some layers might be tweaked or modified by programmer, for some narrower use case.
* If training done with researched & completely new layer, it is useful to be able to handle it as Custom Layer.

## Comparing Model Performance

Python scripts, `ir_perf.py` and `model_perf.py` were written by me to calculate performance differences between TensorFlow model, SSD MobileNet V2 and it's IR. Related files are included in `./benchmark` dir. Brief procedure goes like this:
* OpenCV's `VideoCapture` was used to read each frame and inference was made with each representation of model.
* Time stats recorded included only inference duration until output was retrieved, not including frame preprocessing or loading model from files.
* Average time (ms per frame) for inference was calculated based on each and every frame of the given sample video.
* `capture_frames.py` was written to capture and write frames (in `.jpg`), between pair of video timestamps, during which a person was entering or exiting the frame roughly. This is so to manually determine exact precise (ms) timestamp pairs (start and end) between which a person was visible in frame. Generated `.jpg` file names are the timestamp values in ms.
* Finally in both scripts `ir_perf.py` and `model_perf.py`, total no. of frames (`T`) in which there was a person was counted. And also the no. of wrong inferences (`W`) were detected using timestamp pairs data.
* Confidence threshold of **0.5** was used to classify a wrong inference.
* Finally, accuracy was calculated as, `A = (T - W)/T`.

Following table summarises the stats:<br>
|             | Size(MB)   | Accuracy     | Speed(ms/frame) |
|-------------|------------|--------------|-----------------|
|**Model**    |  69.7      |  0.76        |     49          |
|**IR**       |  33.6      |  0.75        |     18          |

## Model Use Cases

Some of the potential use cases of the people counter app are:
* In places like ATMs, camera positioned at top back of ATM. Results from inference can be assessed to realize any suspicious activity, like spending way too much time inside.
* Determining total number of people in sensitive places, like in a quarantined zone. An alarm/warning can be set to ring if count reaches above a fixed threshold.
* Detecting any intruder in the restricted area. As count data changes after detecting the intruder, securities can be informed by the system automatically.
* To monitor toddlers straying away, not under surveillance anymore, the parents can be notified in their smartphones instantly.
* Automating jobs like door opening when a person approaches the door.

## Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:
* Low light conditions like inference requirements in night time may affect accuracy a bit, but a high precision IR can cope with that. However, high precision models are slower. A decision must be made for the priority of speed and accuracy.
* Another solution for low lighting conditions is a deploying a low light camera with the ability to produce gamma corrected frames to inference upon. But these may affect budget cost for end user.
* Camera's focal length affects angle of view. Usually large focal length cameras are better for outdoor views that can cover a wider area and small focal length cameras are better for indoors. So, if user requires to moniter a wider outdoor view larger focal length will be better. On the other hand smaller focal lenght will be better for conditions to monitor smaller spaces like shop corners etc. Price of the device will also be a factor in this case.

## Model Research

Model used for actual inference was from Intel's Model Zoo of pre-trained models, which is [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html). This was so, because it produced most satisfactory results for the app.

However, in investigating potential people counter models, I tried each of the following three models:

- Model 1: SSD MobileNet V2
  - Link: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
  - Following command was used to convert to IR:<br>
  `python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json -o ./ir`
  - The model was insufficient for the app because it failed to detect boxes at particular time with significant consecutive frames (when person wore dark attire). This produced error in the stats calculation.
  - Efforts were made to improve accuracy by passing `float` as `--data_type` arg as well as specifying `disable_weights_compression` arg, with no noticeable change.
  Lowering probability threshold didn't work either.
  
- Model 2: Faster RCNN Inception V2
  - Link: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
  - Following command was used to convert to IR:<br>
  `python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json -o ./ir`
  - Inference produces error ```Segmentation fault (core dumped)```

- Model 3: SSD Inception V2
  - Link: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
  - Following command was used to convert to IR:<br>
  `python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --transformations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_support.json -o ./ir`
  - The model detected a person accurately, but also tries to detect unnecessary stuff with probability >0.5, which leads to hardship in detecting and counting only humans.
  - Adjusting the probability threshold to multiple values produced other negative side effects, so it wasn't sufficiently satisfactory for the app.

from openvino.inference_engine import IECore
import cv2
import numpy as np

def CheckCanUseDevice(Name):
    if Name in IECore().available_devices:
        return Name
    else:
        print('[Warning] Unavailable device "' + Name + '" has been selected. Use "' + IECore().available_devices[0] + '" instead.')
        return IECore().available_devices[0]

class InferenceData:
    def __init__(self, input_name, input_shape, out_name, out_shape, exec_net):
        self.input_name  = input_name 
        self.input_shape = input_shape
        self.out_name    = out_name   
        self.out_shape   = out_shape  
        self.exec_net    = exec_net   

    def PreProcessInferenceData(self, frame):
        try:
            exec_frame = cv2.resize(frame, (self.input_shape[3], self.input_shape[2]))
            exec_frame = exec_frame.transpose((2, 0, 1))
            exec_frame = exec_frame.reshape(self.input_shape)
        except:
            print("[Error] File read error")
            exec_frame = cv2.resize(np.full((256, 256, 3), (37, 37, 37),np.uint8), (self.input_shape[3], self.input_shape[2])).transpose((2, 0, 1)).reshape(self.input_shape)
        
        return exec_frame

    def InferenceData(self, frame):       
        return self.exec_net.infer(inputs={self.input_name: self.PreProcessInferenceData(frame)})[self.out_name]

    def StartInferenceAsync(self, frame):
        return self.exec_net.start_async(request_id=0, inputs={self.input_name: self.PreProcessInferenceData(frame)})

    def GetInferenceDataAsync(self):
        return self.exec_net.requests[0].output_blobs[self.out_name].buffer

class SetupInference:
    def __init__(self, device_name="AUTO"):
        self.SetValue(device_name)
        pass

    def SetValue(self, device_name="AUTO"):
        self.ie_core = IECore()
        self.device = device_name
       
    def SetupModel(self,  model_path_without_extension, device_name="AUTO"):
        net  = self.ie_core.read_network(model= model_path_without_extension + '.xml', weights= model_path_without_extension + '.bin')
        input_name  = next(iter(net.input_info))
        input_shape = net.input_info[input_name].tensor_desc.dims
        out_name    = next(iter(net.outputs))
        out_shape   = net.outputs[out_name].shape
        
        try:
            exec_net = self.ie_core.load_network(network=net, device_name=device_name, num_requests=1)  
        except RuntimeError:
            print('[Error] Can not setup device(' + device_name + '). Using AUTO.')
            try:
                exec_net = self.ie_core.load_network(network=net, device_name='AUTO', num_requests=1)
            except RuntimeError:
                print('[Error] Can not setup device(AUTO). Using CPU.')
                try:
                    exec_net = self.ie_core.load_network(network=net, device_name='CPU', num_requests=1)
                except RuntimeError:
                    print('[Error] Can not setup device(CPU). Using MYRIAD.')
                    try:
                        exec_net = self.ie_core.load_network(network=net, device_name='MYRIAD', num_requests=1)
                    except RuntimeError:
                        print('[Error] Can not setup device(MYRIAD). Using GPU.')
                        try:
                            exec_net = self.ie_core.load_network(network=net, device_name='GPU', num_requests=1)
                        except RuntimeError as e:
                            raise ValueError("Device setup failed. (OpenVINO)\n" + str(e))
            
        del net
        return InferenceData(input_name, input_shape, out_name, out_shape, exec_net)

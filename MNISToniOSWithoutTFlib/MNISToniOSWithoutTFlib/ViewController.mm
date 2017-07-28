//
//  ViewController.m
//  MNISToniOSWithoutTFlib
//
//  Created by Li,Yan(MMS) on 2017/7/17.
//  Copyright © 2017年 Li,Yan(MMS). All rights reserved.
//

#import "ViewController.h"
#import "MNISToniOSWithoutTFlib-Swift.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
@interface ViewController ()
// ui
@property (nonatomic, strong) DrawView *drawView;
@property (nonatomic, strong) UILabel *lable;

// metal
@property(nonatomic, strong) MPSCNNConvolutionDescriptor *conv1descriptor;
@property(nonatomic, strong) MPSCNNConvolution *conv1layer;
@property(nonatomic, strong) MPSImageDescriptor *conv1outdescriptor;
@property(nonatomic, strong) MPSCNNPoolingMax *pool1layer;
@property(nonatomic, strong) MPSImageDescriptor *pool1outdescriptor;
@property(nonatomic, strong) MPSCNNConvolutionDescriptor *conv2descriptor;
@property(nonatomic, strong) MPSCNNConvolution *conv2layer;
@property(nonatomic, strong) MPSImageDescriptor *conv2outdescriptor;
@property(nonatomic, strong) MPSCNNPoolingMax *pool2layer;
@property(nonatomic, strong) MPSImageDescriptor *pool2outdescriptor;
@property(nonatomic, strong) MPSCNNConvolutionDescriptor *fc1descriptor;
@property(nonatomic, strong) MPSCNNFullyConnected *fc1layer;
@property(nonatomic, strong) MPSImageDescriptor *fc1outdescriptor;
@property(nonatomic, strong) MPSCNNConvolutionDescriptor *fc2descriptor;
@property(nonatomic, strong) MPSCNNFullyConnected *fc2layer;
@property(nonatomic, strong) MPSImageDescriptor *fc2outdescriptor;
@property(nonatomic, strong) MPSImageDescriptor *softmaxOutput;
@property(nonatomic, strong) MPSCNNSoftMax *softmaxLayer;
@property(nonatomic, strong) MPSImageDescriptor *inputDescriptor;
@property(nonatomic, strong) NSMutableArray<id<MTLCommandBuffer>> *pendingBuffers;
@property(nonatomic, strong) NSMutableArray<MPSImage *> *results;
@property(nonatomic, strong) id commandQueue;
@property(nonatomic, assign) float *dataArray;

@end

static constexpr int kUsedExamples = 1;
static constexpr int kImageSide = 28;
static constexpr int kOutputs = 10;
static constexpr int kInputLength = kImageSide * kImageSide;
static constexpr int kImageSide2 = kImageSide / 2;
static constexpr int kImageSide4 = kImageSide / 4;
@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    [self setButton];
    NSInteger sWidth = [UIScreen mainScreen].bounds.size.width;
    self.lable = [[UILabel alloc]initWithFrame:CGRectMake(sWidth / 2 - 100, 100, 200, 30 )];
    [self.lable setBackgroundColor:[UIColor whiteColor]];
    self.lable.textAlignment = NSTextAlignmentCenter;
    self.lable.text = @"Write A Number";
    [self.view addSubview:self.lable];
    
    self.view.backgroundColor = [UIColor colorWithRed:240/255.0 green:240/255.0 blue:240/255.0 alpha:0.2];
}

- (void) viewWillAppear:(BOOL)animated {
    
}

-(void) viewDidAppear:(BOOL)animated {
    NSInteger sWidth = [UIScreen mainScreen].bounds.size.width;
    self.drawView = [[DrawView alloc] initWithFrame:CGRectMake((sWidth - 300) / 2, 150, 300, 300)];
    [self.view addSubview:self.drawView];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        NSLog(@"no metal support");
        return;
    }
    [self initMetal:device];
}

-(void) setButton {
    NSInteger sWidth = [UIScreen mainScreen].bounds.size.width;
    
    UIButton *clearButton = [[UIButton alloc]initWithFrame:CGRectMake(10, 500, sWidth / 3 - 20, 30)];
    clearButton.backgroundColor = [UIColor whiteColor];
    [clearButton setTitle:@"clear" forState:UIControlStateNormal];
    clearButton.titleLabel.textColor = [UIColor redColor];
    [clearButton addTarget:self action:@selector(buttonAction:) forControlEvents:UIControlEventTouchUpInside];
    [clearButton setTitleColor:[UIColor redColor] forState:UIControlStateNormal];
    clearButton.tag = 1001;
    [self.view addSubview:clearButton];
    
    UIButton *undoButton = [[UIButton alloc]initWithFrame:CGRectMake(sWidth / 3 + 10, 500, sWidth / 3 - 20, 30)];
    undoButton.backgroundColor = [UIColor whiteColor];
    [undoButton setTitle:@"undo" forState:UIControlStateNormal];
    [undoButton addTarget:self action:@selector(buttonAction:) forControlEvents:UIControlEventTouchUpInside];
    [undoButton setTitleColor:[UIColor redColor] forState:UIControlStateNormal];
    undoButton.tag = 1002;
    [self.view addSubview:undoButton];
    
    UIButton *predictButton = [[UIButton alloc]initWithFrame:CGRectMake(sWidth * 2 / 3 + 10, 500, sWidth / 3 - 20, 30)];
    predictButton.backgroundColor = [UIColor whiteColor];
    [predictButton setTitle:@"predict" forState:UIControlStateNormal];
    [predictButton addTarget:self action:@selector(buttonAction:) forControlEvents:UIControlEventTouchUpInside];
    [predictButton setTitleColor:[UIColor redColor] forState:UIControlStateNormal];
    predictButton.tag = 1003;
    [self.view addSubview:predictButton];
}



-(void)buttonAction:(UIButton *)sender {
    switch (sender.tag) {
        case 1001:
            [self.drawView clear];
            break;
        case 1002:
            [self.drawView undo];
            break;
        case 1003:
            [self predictWithImage:[self.drawView getImage]];
            break;
        default:
            break;
    }
}

-(void) setPredictLabel:(NSInteger) result {
    if (self.lable) {
        self.lable.text = [NSString stringWithFormat:@"Result: %ld",(long)result];
    }
}

#pragma mark - init metal descriptor
-(void) initMetal:(id) nDevice {
    float *conv1weights = loadTensor(@"W_conv1", 5 * 5 * 1 * 32);
    float *conv1biases = loadTensor(@"b_conv1", 32);
    float *conv2weights = loadTensor(@"W_conv2", 5 * 5 * 32 * 64);
    float *conv2biases = loadTensor(@"b_conv2", 64);
    float *fc1weights = loadTensor(@"W_fc1", 7 * 7 * 64 * 1024);
    float *fc1biases = loadTensor(@"b_fc1", 1024);
    float *fc2weights = loadTensor(@"W_fc2", 1024 * 10);
    float *fc2biases = loadTensor(@"b_fc2", 10);
    
    id<MTLDevice> device = nDevice;

    const MPSCNNNeuronReLU *reluUnit = [[MPSCNNNeuronReLU alloc] initWithDevice:device a:0];
    
    self.conv1descriptor = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:5 kernelHeight:5 inputFeatureChannels:1 outputFeatureChannels:32 neuronFilter:reluUnit];
    self.conv1layer = [[MPSCNNConvolution alloc] initWithDevice:device convolutionDescriptor:self.conv1descriptor kernelWeights:conv1weights biasTerms:conv1biases flags:MPSCNNConvolutionFlagsNone];
    self.conv1outdescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16 width:kImageSide height:kImageSide featureChannels:32];
    
    self.pool1layer = [[MPSCNNPoolingMax alloc]initWithDevice:device kernelWidth:2 kernelHeight:2 strideInPixelsX:2 strideInPixelsY:2];
    self.pool1layer.offset = (MPSOffset){1, 1, 0};
    self.pool1layer.edgeMode = MPSImageEdgeModeClamp;
    self.pool1outdescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16 width:kImageSide2 height:kImageSide2 featureChannels:32];
    
    self.conv2descriptor = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:5 kernelHeight:5 inputFeatureChannels:32 outputFeatureChannels:64 neuronFilter:reluUnit];
    self.conv2layer = [[MPSCNNConvolution alloc] initWithDevice:device convolutionDescriptor:self.conv2descriptor kernelWeights:conv2weights biasTerms:conv2biases flags:MPSCNNConvolutionFlagsNone];
    self.conv2outdescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16 width:kImageSide2 height:kImageSide2 featureChannels:64];
    self.pool2layer = [[MPSCNNPoolingMax alloc] initWithDevice:device kernelWidth:2 kernelHeight:2 strideInPixelsX:2 strideInPixelsY:2];
    self.pool2layer.offset = (MPSOffset){1, 1, 0};
    self.pool2layer.edgeMode = MPSImageEdgeModeClamp;
    self.pool2outdescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16 width:kImageSide4 height:kImageSide4 featureChannels:64];
    
    self.fc1descriptor = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:kImageSide4 kernelHeight:kImageSide4 inputFeatureChannels:64 outputFeatureChannels:1024 neuronFilter:reluUnit];
    self.fc1layer = [[MPSCNNFullyConnected alloc] initWithDevice:device convolutionDescriptor:self.fc1descriptor kernelWeights:fc1weights biasTerms:fc1biases flags:MPSCNNConvolutionFlagsNone];
    self.fc1outdescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16 width:1 height:1 featureChannels:1024];
    
    self.fc2descriptor = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:1 kernelHeight:1 inputFeatureChannels:1024 outputFeatureChannels:kOutputs neuronFilter:nil];
    self.fc2layer = [[MPSCNNFullyConnected alloc] initWithDevice:device convolutionDescriptor:self.fc2descriptor kernelWeights:fc2weights biasTerms:fc2biases flags:MPSCNNConvolutionFlagsNone];
    self.fc2outdescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16 width:1 height:1 featureChannels:kOutputs];
    self.softmaxOutput = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16 width:1 height:1 featureChannels:kOutputs];
    self.softmaxLayer = [[MPSCNNSoftMax alloc]initWithDevice:device];
    self.inputDescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32 width:kImageSide height:kImageSide featureChannels:1];
    
    self.pendingBuffers = [[NSMutableArray alloc] init];
    self.results = [[NSMutableArray alloc] init];
    
}

-(void) predictWithImage:(UIImage *)drawedImage {
    UIImage *scaledImage = [self scaleImage:drawedImage];
    UIImage *image = [self convertImageToGrayScale:scaledImage];

    float *data = [self getGrayPixelFromImage:image atX:0 andY:0 count:kInputLength];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        NSLog(@"no metal support");
        return;
    }
    
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    MPSImage *inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:self.inputDescriptor];
    [inputImage.texture replaceRegion:MTLRegionMake2D(0, 0, kImageSide, kImageSide) mipmapLevel:0 withBytes:data bytesPerRow:sizeof(float) * kImageSide];
    [MPSTemporaryImage prefetchStorageWithCommandBuffer:buffer imageDescriptorList:@[self.conv1outdescriptor, self.pool1outdescriptor, self.conv2outdescriptor, self.pool2outdescriptor, self.fc1outdescriptor, self.fc2outdescriptor]];
    
    MPSTemporaryImage *c1o = [MPSTemporaryImage temporaryImageWithCommandBuffer:buffer imageDescriptor:self.conv1outdescriptor];
    [self.conv1layer encodeToCommandBuffer:buffer sourceImage:inputImage destinationImage:c1o];
    
    MPSTemporaryImage *p1o = [MPSTemporaryImage temporaryImageWithCommandBuffer:buffer imageDescriptor:self.pool1outdescriptor];
    [self.pool1layer encodeToCommandBuffer:buffer sourceImage:c1o destinationImage:p1o];
    
    MPSTemporaryImage *c2o = [MPSTemporaryImage temporaryImageWithCommandBuffer:buffer imageDescriptor:self.conv2outdescriptor];
    [self.conv2layer encodeToCommandBuffer:buffer sourceImage:p1o destinationImage:c2o];
    
    MPSTemporaryImage *p2o = [MPSTemporaryImage temporaryImageWithCommandBuffer:buffer imageDescriptor:self.pool2outdescriptor];
    [self.pool2layer encodeToCommandBuffer:buffer sourceImage:c2o destinationImage:p2o];
    
    MPSTemporaryImage *fc1tdi = [MPSTemporaryImage temporaryImageWithCommandBuffer:buffer imageDescriptor:self.fc1outdescriptor];
    [self.fc1layer encodeToCommandBuffer:buffer sourceImage:p2o destinationImage:fc1tdi];
    
    MPSTemporaryImage *fc2tdi = [MPSTemporaryImage temporaryImageWithCommandBuffer:buffer imageDescriptor:self.fc2outdescriptor];
    [self.fc2layer encodeToCommandBuffer:buffer sourceImage:fc1tdi destinationImage:fc2tdi];
    
    __block MPSImage *resultImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:self.softmaxOutput];
    [self.softmaxLayer encodeToCommandBuffer:buffer sourceImage:fc2tdi destinationImage:resultImage];
    
    [buffer commit];
    [buffer waitUntilCompleted];
    
    const auto start = CACurrentMediaTime();
    

    const size_t numSlices = (resultImage.featureChannels + 3)/4;
    float16_t halfs[numSlices * 4];
    NSLog(@"size of float16_t %lu",sizeof(float16_t));
    for (size_t i = 0; i < numSlices; i += 1) {
        [resultImage.texture getBytes:&halfs[i * 4] bytesPerRow:8 bytesPerImage:8 fromRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1) mipmapLevel:0 slice:i];
        for (size_t j = i * 4; j < i * 4 + 4; j++) {
            NSLog(@"half %zu %f", j, halfs[j]);
        }
    }
    
    float results[kOutputs];
    
    vImage_Buffer fullResultVImagebuf;
    fullResultVImagebuf.data = results;
    fullResultVImagebuf.height = 1;
    fullResultVImagebuf.width = kOutputs;
    fullResultVImagebuf.rowBytes = kOutputs * 4;
    
    vImage_Buffer halfResultVImagebuf;
    halfResultVImagebuf.data = halfs;
    halfResultVImagebuf.height = 1;
    halfResultVImagebuf.width = kOutputs;
    halfResultVImagebuf.rowBytes = kOutputs * 2;
    
    vImageConvert_Planar16FtoPlanarF(&halfResultVImagebuf, &fullResultVImagebuf, 0);
    
    int bestIndex = -1;
    float bestProbability = 0;
    for (auto i = 0; i < kOutputs; i++) {
        const auto probability = results[i];
        if (probability > bestProbability) {
            bestProbability = probability;
            bestIndex = i;
        }
    }
    
    NSLog(@"lydebug predict %d",bestIndex);
    [self setPredictLabel:(NSInteger) bestIndex];
    
    NSLog(@"Time: %g seconds", CACurrentMediaTime() - start);
}



static float *loadTensor(NSString *baseName, NSUInteger length) {
    NSString *path = [[NSBundle mainBundle] pathForResource:baseName ofType:nil];
    NSData *data = [NSData dataWithContentsOfFile:path];
    float *tensor = new float[length];
    for (NSUInteger i = 0; i < length; i++) {
        [data getBytes:&tensor[i] range:NSMakeRange(i * sizeof(float), sizeof(float))];
    }
    return tensor;
}

#pragma mark - process image for input


- (float *)getGrayPixelFromImage:(UIImage*)image atX:(int)x andY:(int)y count:(int)count
{
    // First get the image into your data buffer
    CGImageRef imageRef = [image CGImage];
    NSUInteger width = CGImageGetWidth(imageRef);
    NSUInteger height = CGImageGetHeight(imageRef);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    unsigned char *rawData = (unsigned char*) calloc(height * width * 4, sizeof(unsigned char));
    NSUInteger bytesPerPixel = 4;
    NSUInteger bytesPerRow = bytesPerPixel * width;
    NSUInteger bitsPerComponent = 8;
    CGContextRef context = CGBitmapContextCreate(rawData, width, height,
                                                 bitsPerComponent, bytesPerRow, colorSpace,
                                                 kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(colorSpace);
    
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), imageRef);
    CGContextRelease(context);
    
    // Now your rawData contains the image data in the RGBA8888 pixel format.
    NSUInteger byteIndex = (bytesPerRow * y) + x * bytesPerPixel;
//    float *array = new float(count + 1);
    if (self.dataArray != nullptr) {
        free(self.dataArray);
        
    }
    self.dataArray = new float[count];
    for (int i = 0 ; i < count ; ++i)
    {
        //CGFloat alpha = ((CGFloat) rawData[byteIndex + 3] ) / 255.0f;
        CGFloat red   = ((CGFloat) rawData[byteIndex]     ) ;
        //CGFloat green = ((CGFloat) rawData[byteIndex + 1] ) ;
        //CGFloat blue  = ((CGFloat) rawData[byteIndex + 2] ) ;
        byteIndex += bytesPerPixel;
        self.dataArray[i] = (255 - red) / 255.0f;
    }
    
    free(rawData);
    
    return self.dataArray;
}

- (UIImage *)convertImageToGrayScale:(UIImage *)image {
    // Create image rectangle with current image width/height
    CGRect imageRect = CGRectMake(0, 0, image.size.width, image.size.height);
    
    // Grayscale color space
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    
    // Create bitmap content with current image size and grayscale colorspace
    CGContextRef context = CGBitmapContextCreate(nil, image.size.width, image.size.height, 8, 0, colorSpace, kCGImageAlphaNone);
    
    // Draw image into current context, with specified rectangle
    // using previously defined context (with grayscale colorspace)
    CGContextDrawImage(context, imageRect, [image CGImage]);
    
    // Create bitmap image info from pixel data in current context
    CGImageRef imageRef = CGBitmapContextCreateImage(context);
    
    // Create a new UIImage object
    UIImage *newImage = [UIImage imageWithCGImage:imageRef];
    
    // Release colorspace, context and bitmap information
    CGColorSpaceRelease(colorSpace);
    CGContextRelease(context);
    CFRelease(imageRef);
    
    // Return the new grayscale image
    return newImage;
}

-(UIImage *)scaleImage:(UIImage *)image {
    CGSize oldSize = image.size;
    NSLog(@"before scale w %f, y %f",oldSize.width, oldSize.height);
    CGFloat nHeight = 0.0;
    CGFloat nWidth = 0.0;
    if (oldSize.height > oldSize.width) {
        nHeight = 28;
        nWidth = nHeight * (oldSize.width / oldSize.height);
    } else {
        nWidth = 28;
        nHeight = nWidth * (oldSize.height / oldSize.width);
    }
    
    CGRect rect = CGRectMake(0,0,nHeight,nWidth);
    CGRect rectfull = CGRectMake(0,0,kImageSide,kImageSide);
    UIGraphicsBeginImageContext( rect.size );
    
    UIImage *background = [self imageFromColor:[UIColor whiteColor]];
    [background drawInRect:rectfull];
    [image drawInRect:rect];
    
    UIImage *picture1 = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    NSData *imageData = UIImagePNGRepresentation(picture1);
    UIImage *img=[UIImage imageWithData:imageData];
    NSLog(@"after scale w %f, y %f",img.size.width, img.size.height);
    
    return img;
}


- (UIImage *)imageFromColor:(UIColor *)color {
    CGRect rect = CGRectMake(0, 0, 1, 1);
    UIGraphicsBeginImageContext(rect.size);
    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextSetFillColorWithColor(context, [color CGColor]);
    CGContextFillRect(context, rect);
    UIImage *image = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return image;
}

@end

//
//  DrawView.swift
//  MNISTTFiOS
//
//  Created by Li,Yan(MMS) on 2017/6/27.
//  Copyright © 2017年 Li,Yan(MMS). All rights reserved.
//

import UIKit

class DrawView: UIView {

    var allPathPoints:NSMutableArray = []
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        self.isUserInteractionEnabled = true;
        self.backgroundColor = UIColor.white
    }
    
    required init?(coder aDecoder: NSCoder) {
        super.init(coder: aDecoder)
        fatalError("init(coder:) has not been implemented")
    }
    
    func clear() {
        allPathPoints.removeAllObjects()
        self.setNeedsDisplay()
    }
    
    func undo() {
        allPathPoints.removeLastObject()
        self.setNeedsDisplay()
    }
    
    func getImage() -> UIImage! {
        UIGraphicsBeginImageContextWithOptions(self.frame.size, true, self.layer.contentsScale)
        self.layer.render(in: UIGraphicsGetCurrentContext()!)
        let image = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return image
    }
    
    func buttonClick(sender: UIButton) {

    }
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        let touch = touches.first!
        let startPoint = touch.location(in: touch.view)
        let points = NSMutableArray()
        points.add(NSValue(cgPoint: startPoint))
        allPathPoints.add(points)
        self.setNeedsDisplay()
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        let touch = touches.first!
        let movePoint = touch.location(in: touch.view)
        let pathPoints = allPathPoints.lastObject as! NSMutableArray
        pathPoints.add(NSValue(cgPoint: movePoint))
        self.setNeedsDisplay()
    }
    
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        self.touchesMoved(touches, with: event)
    }
    
    override func draw(_ rect: CGRect) {
        let context = UIGraphicsGetCurrentContext()
        for case let pathPoints as NSArray in allPathPoints {
            for i in 0...pathPoints.count - 1  {
                let point = (pathPoints[i] as! NSValue).cgPointValue
                if i == 0 {
                    context?.move(to: point)
                } else {
                    context?.addLine(to: point)
                }
            }
        }
        context?.setStrokeColor(UIColor.black.cgColor)
        context?.setLineCap(.round)
        context?.setLineJoin(.round)
        context?.setLineWidth(4.0)
        context?.strokePath()
    }

}

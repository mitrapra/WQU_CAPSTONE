//
//  Item.swift
//  WQU_Capstone
//
//  Created by PRATANU MITRA on 3/31/24.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}

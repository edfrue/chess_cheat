import numpy as np
import cv2
from PIL import ImageGrab
class boardParser:
    _piecesFEN = '--RNBQKPrnbqkp'
    PLAYING_COLOR_WHITE = 0
    PLAYING_COLOR_BLACK = 1
    def __init__(self):
        self.parsedPieces = None
        self.parsedMasks = None
        self.lines = None
        self.corners = None
        self.myColor = self.PLAYING_COLOR_WHITE
        self.upperLeft = None
        self.lowerLeft = None
        self.upperRight = None
        self.squareSize = None
        self.setupImg = None
        self.filterOut = None

    @staticmethod
    def findHorzVertLines(edges, dRes = 1, phiRes = 1, thresh = 20):
        ''' Find horizontal and vertical Lines
        '''
        lines = cv2.HoughLines(edges,dRes,phiRes*np.pi/180, thresh)
        lines = lines.squeeze()
        linesVert = lines[np.where(lines[:,1] < np.pi/180)]
        linesVert = linesVert[:,0]
        linesVert.sort()
        linesHorz = lines[np.where(lines[:,1] > np.pi/2 -np.pi/180)]
        linesHorz = linesHorz[np.where(linesHorz[:,1] < np.pi/2 + np.pi/180)]
        linesHorz = linesHorz[:,0]
        linesHorz.sort()
        return (linesHorz, linesVert)    

    def plotGridLines(self):
        ''' Plot horizontal and vertical Lines
        '''
        grid = self.setupImg.copy()
        #Plot Horz Lines
        for y in self.lines[0]:
            cv2.line(grid,(0,y),(grid.shape[1], y),(0,255,0),2)
        #Plot Vert Lines
        for x in self.lines[1]:
            cv2.line(grid,(x,0),(x,grid.shape[0]),(0,255,0),2)
        cv2.imshow('Grid',grid)

    @staticmethod
    def mostFrequentDist(linesHorz, linesVert, minDist = 20, res = 300):
        ''' Find most frequent distance between horizontal and vertical Lines
        '''
        a,b = np.meshgrid(linesHorz, linesHorz)
        dy = np.abs(a-b)
        a,b = np.meshgrid(linesVert, linesVert)
        dx = np.abs(a-b)
        d = np.hstack((dy.reshape(-1),dx.reshape(-1)))
        d.sort()
        d = d[np.argmax(d > minDist):]
        hist, histBins  = np.histogram(d, res)
        # most3Bins = np.argsort(hist)[-3:][::-1]
        most = np.argmax(hist)
        horz = []
        vert = []
        for i in range(dy.shape[0]):
            hist,_ = np.histogram(dy[i,:], res, (histBins.min(), histBins.max()))
            # if np.sum(hist[most3Bins]) > 2:
            if hist[most] > 0:
                horz.append(i)
        for i in range(dx.shape[0]):
            hist,_ = np.histogram(dx[i,:], res, (histBins.min(), histBins.max()))
            # if np.sum(hist[most3Bins]) > 2:
            if hist[most] > 0:
                vert.append(i)
        return (linesHorz[horz], linesVert[vert])

    @staticmethod
    def findCorners(xCorner, yCorner):
        ''' Find Intersections between horz and vert Lines
        '''
        mg = np.meshgrid(xCorner, yCorner)
        corners = np.vstack((mg[0].reshape(-1), mg[1].reshape(-1))).T
        return corners.astype(int)

    @staticmethod
    def findValidCorners(edges, corners, res = 5, score = 1.5):    
        ''' Find Corners which extend horizontally and vertically in edge Img
            ╋
        '''
        validCornersInd = []
        for i in range(corners.shape[0]):
            Y1 = int(max(corners[i,0]-res, 0))
            Y2 = int(min(corners[i,0]+res+1, edges.shape[0]))
            X1 = int(max(corners[i,1]-res, 0))
            X2 = int(min(corners[i,1]+res+1, edges.shape[1]))
            scoreX = np.sum(edges[corners[i,0],X1:X2])
            scoreY = np.sum(edges[Y1:Y2,corners[i,1]])
            if scoreX > score*res*255 and scoreY > score*res*255:
                validCornersInd.append(i)
        return corners[validCornersInd]

    def findValidCorners1(edges, corners, res, maxScoreDiff):
        validCornersInd = []
        for i in range(corners.shape[0]):
            Y1 = int(corners[i,0]-res)
            Y2 = int(corners[i,0]+res+1)
            X1 = int(corners[i,1]-res)
            X2 = int(corners[i,1]+res+1)
            if Y1 > 20 and Y2 < edges.shape[0] - 20 and X1 > 20 and X2 < edges.shape[1] - 20:
                scoreUL = np.sum(edges[Y1:corners[i,0]+1,X1:corners[i,1]+1])
                scoreLR = np.sum(edges[corners[i,0]:Y2,corners[i,1]:X2])
                scoreUR = np.sum(edges[Y1:corners[i,0]+1,corners[i,1]:X2])
                scoreLL = np.sum(edges[corners[i,0]:Y2,X1:corners[i,1]+1])
                if np.abs(scoreUL - scoreLR) < maxScoreDiff and np.abs(scoreUR - scoreLL) < maxScoreDiff:
                    validCornersInd.append(i)
        return corners[validCornersInd]
    
    def findChessBoard(gray, kSize = 7, thresh1 = 10, thresh2 = 50):
        kernelUpperLeft = np.ones((2*kSize, 2*kSize), dtype='int32')
        kernelUpperLeft[kSize:, :] = 0
        kernelUpperLeft[:, kSize:] = 0
        kernelLowerLeft = kernelUpperLeft[::-1,:]
        kernelUpperRight = kernelUpperLeft[:, ::-1]
        kernelLowerRight = kernelUpperLeft[::-1, ::-1]
        kernel1 = kernelUpperLeft - kernelLowerRight
        kernel2 = kernelUpperRight - kernelLowerLeft
        kernel3 = kernelUpperLeft - kernelUpperRight
        kernel4 = kernelLowerLeft - kernelLowerRight
        kernel5 = kernelUpperLeft - kernelLowerLeft
        kernel6 = kernelUpperRight - kernelLowerRight
        g = gray.astype('int32')
        filtered1 = cv2.filter2D(gray, cv2.CV_32S, kernel1)
        filtered2 = cv2.filter2D(gray, cv2.CV_32S, kernel2)
        filtered3 = cv2.filter2D(gray, cv2.CV_32S, kernel3)
        filtered4 = cv2.filter2D(gray, cv2.CV_32S, kernel4)
        filtered5 = cv2.filter2D(gray, cv2.CV_32S, kernel5)
        filtered6 = cv2.filter2D(gray, cv2.CV_32S, kernel6)
        filtList = [filtered1, filtered2, filtered3, filtered4, filtered5, filtered6]
        f = 255*np.ones_like(gray)
        normalizedList = [np.abs(filt)*255/np.abs(filt).max() for filt in filtList]
        f[normalizedList[0] > thresh1] = 0
        f[normalizedList[1] > thresh1] = 0
        f[normalizedList[2] < thresh2] = 0
        f[normalizedList[3] < thresh2] = 0
        f[normalizedList[4] < thresh2] = 0
        f[normalizedList[5] < thresh2] = 0
        return f
        
    def plotFilterOut(self):
        cv2.imshow('Filter Output', self.filterOut)
        

    def plotCorners(self):
        cornerImg = self.setupImg.copy()
        cornerImg[self.corners[1,:], self.corners[0,:]] = (0,255,0)
        for p in self.corners:
            cv2.circle(cornerImg, (p[1], p[0]), 1, (0,255,0),2)
        cv2.imshow('Intersections', cornerImg)

    @staticmethod
    def getSquares(img, upperLeft, squareSize):
        ''' Returns 64 chess squares
        '''
        sizeY = squareSize
        sizeX = squareSize
        squares = []
        for i in range(8):
            for j in range(8):
                top = upperLeft[0]+i*sizeY + 1 # +1? magic value
                bottom = top + sizeY
                left = upperLeft[1]+j*sizeX + 1 # +1? magic value
                right = left + sizeX
                squares.append(img[top:bottom, left:right])
        return squares

    @staticmethod
    def maskPiece(square, backGround):
        ''' Segmentation of chess Piece from Background
        '''
        squareDiff = np.abs(square.astype('int32') - backGround.astype('int32'))
        squareDiffMask = np.ones_like(square)*255
        squareDiffMask[squareDiff < 30] = 0
        kernel = np.ones((1,1),np.uint8)
        squareDiffMask = cv2.morphologyEx(squareDiffMask,cv2.MORPH_OPEN,kernel)
        _, contours, _ = cv2.findContours(squareDiffMask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(square)
        c = max(contours, key = np.shape)
        cv2.drawContours(mask,[c],0,255,-1)
        out = np.zeros_like(mask)
        out[mask == 255] = square[mask == 255]
        out[squareDiff < 5] = 0
        mask[squareDiff < 5] = 0
        return [out, mask]

    def extractPieces(self, squares):
        ''' Extract piece masks from default board start positions
        Takes 64 ordered squares (default board start positions)
        Returns piece masks in Order: '--RNBQKPrnbqkp'
        FEN Notation, -- is empty 1 white empty square and one black empty square
        '''
        black = [self.maskPiece(sq, bg) for sq, bg in zip(squares[0:16], self.backGroundColors[0:16])]
        white = [self.maskPiece(sq, bg) for sq, bg  in zip(squares[-16:], self.backGroundColors[-16:])]
        blackPiece = [p.astype('float64') for p,m in black]
        blackMask = [m.astype('bool') for p,m in black]
        whitePiece = [p.astype('float64') for p,m in white]
        whiteMask = [m.astype('bool') for p,m in white]
        # black Rook
        bRook = (blackPiece[0] + blackPiece[7])/2
        bMaskRook = (blackMask[0] + blackMask[7])
        # black Knight
        bKnight = (blackPiece[1] + blackPiece[6])/2
        bMaskKnight = (blackMask[1] + blackMask[6])
        # black Bishop
        bBishop = (blackPiece[2] + blackPiece[5])/2
        bMaskBishop = (blackMask[2] + blackMask[5])
        # black Queen
        bQueen = blackPiece[3]
        bMaskQueen = blackMask[3]
        # black King
        bKing = blackPiece[4]
        bMaskKing = blackMask[4]
        # black Pawn
        bPawn = sum(blackPiece[8:16])/8
        bMaskPawn = np.prod(np.array(blackMask[8:16]),0).astype('bool')
        # black empty square
        bEmpty = self.blackBG
        bMaskEmpty = np.ones_like(bEmpty).astype('bool')
        # white Rook
        wRook = (whitePiece[-1] + whitePiece[-8])/2
        wMaskRook = (whiteMask[-1] + whiteMask[-8])
        # white Knight
        wKnight = (whitePiece[-2] + whitePiece[-7])/2
        wMaskKnight = (whiteMask[-2] + whiteMask[-7])
        # white Bishop
        wBishop = (whitePiece[-3] + whitePiece[-6])/2
        wMaskBishop = (whiteMask[-3] + whiteMask[-6])
        # white Queen
        wQueen = whitePiece[-5]
        wMaskQueen = whiteMask[-5]
        # white King
        wKing = whitePiece[-4]
        wMaskKing = whiteMask[-4]
        # white Pawn
        wPawn = sum(whitePiece[-16:-8])/8
        wMaskPawn = np.prod(np.array(whiteMask[-16:-8]),0).astype('bool')
        # blackPiece empty square
        wEmpty = self.whiteBG
        wMaskEmpty = np.ones_like(wEmpty).astype('bool')
        pieceList = [wEmpty.astype('int32'), bEmpty.astype('int32'), wRook.astype('int32'), wKnight.astype('int32'), wBishop.astype('int32'), wQueen.astype('int32'), wKing.astype('int32'), wPawn.astype('int32'), bRook.astype('int32'), bKnight.astype('int32'), bBishop.astype('int32'), bQueen.astype('int32'), bKing.astype('int32'), bPawn.astype('int32')]
        maskList = [wMaskEmpty.astype('int32'), bMaskEmpty.astype('int32'), wMaskRook.astype('int32'), wMaskKnight.astype('int32'), wMaskBishop.astype('int32'), wMaskQueen.astype('int32'), wMaskKing.astype('int32'), wMaskPawn.astype('int32'), bMaskRook.astype('int32'), bMaskKnight.astype('int32'), bMaskBishop.astype('int32'), bMaskQueen.astype('int32'), bMaskKing.astype('int32'), bMaskPawn.astype('int32')]
        return [pieceList, maskList]

    @staticmethod
    def diffScore(square, piece, mask):
        ''' Squared differences at masked area
        '''
        maskedSquare = mask*square
        return np.sum((maskedSquare.astype('int32')-piece)**2)

    def findBestMatch(self, square, pieceList, maskList):
        ''' finds the piece that matches best for the square
        simple difference matching
        '''
        sums = [self.diffScore(square, p, m) for p,m in zip(pieceList, maskList)]
        return np.argmin(sums)

    def learnPieces(self, img):
        ''' Find chess pieces for template matching 
        takes Input Image with default chess starting positions
        '''
        self.setupImg = img
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        # self.edges = cv2.Canny(gray,100,150)
        # linesHorz, linesVert = self.findHorzVertLines(self.edges)
        # # linesHorz, linesVert = self.mostFrequentDist(linesHorz, linesVert, 3)
        # self.lines = (linesHorz, linesVert)
        # corners = self.findCorners(linesHorz, linesVert)
        # self.corners = self.findValidCorners(self.edges, corners)
        
        # filterOut = boardParser.findChessBoard(gray, 11, 10, 50)
        # filterOut = boardParser.findChessBoard(gray, 11, 7, 30)
        # filterOut[filterOut>0] = 255
        
        self.filterOut = boardParser.findChessBoard(gray, 22, 50, 30)
        filterOut1 = boardParser.findChessBoard(gray, 5, 20, 30)
        self.filterOut[filterOut1 == 0] = 0
        linesHorz, linesVert = boardParser.findHorzVertLines(self.filterOut, 1, 1, 2)
        self.lines = boardParser.mostFrequentDist(linesHorz, linesVert, 30, int(min(gray.shape)/4))
        self.corners = self.findCorners(self.lines[0], self.lines[1])
        self.squareSize = int(np.round((self.corners[:,0].max() - self.corners[:,0].min())/6))
        self.upperLeft = (self.corners[:,0].min() - self.squareSize, self.corners[:,1].min() - self.squareSize)
        self.upperRight = (self.corners[:,0].min() - self.squareSize, self.corners[:,1].max() + self.squareSize)
        self.lowerLeft = (self.corners[:,0].max() + self.squareSize, self.corners[:,1].min()- self.squareSize)        
        self.squares = self.getSquares(gray, self.upperLeft, self.squareSize)
        self.whiteBG = np.ones_like(self.squares[0])*np.mean(self.squares[16:24:2]).astype('uint8')
        self.blackBG = np.ones_like(self.squares[0])*np.mean(self.squares[17:24:2]).astype('uint8')
        bg = [self.whiteBG, self.blackBG] * 4 + [self.blackBG, self.whiteBG] * 4
        self.backGroundColors = bg * 4
        self.parsedPieces, self.parsedMasks = self.extractPieces(self.squares)

    def getPosition(self, gray):
        ''' writes out board position in FEN Notaion
        '''
        squares = self.getSquares(gray, self.upperLeft, self.squareSize)
        fen = []
        consecBlank = 0
        for i in range(64):
            if i % 8 == 0 and i != 0:
                if consecBlank != 0:
                    fen.append(str(consecBlank))
                    consecBlank = 0
                fen.append('/')
            f = self._piecesFEN[self.findBestMatch(squares[i], self.parsedPieces, self.parsedMasks)]
            if f == '-':
                consecBlank += 1
            else:
                if consecBlank != 0:
                    fen.append(str(consecBlank))
                    consecBlank = 0
                fen.append(f)
        if consecBlank != 0: ## last row fix missing empty square
            fen.append(str(consecBlank))
            consecBlank = 0
        if self.myColor == self.PLAYING_COLOR_BLACK:
            fen.reverse()    
        return ''.join(fen)

    def findPlayingColour(self, gray):
        squares = self.getSquares(gray, self.upperLeft, self.squareSize)
        if np.sum(squares[0]) < np.sum(squares[-1]):
            self.myColor = self.PLAYING_COLOR_WHITE
        else:
            self.myColor = self.PLAYING_COLOR_BLACK

    def getSquarePosition(self, squareIndex):
        ''' Returns Squares Position in Pixels on Screen from Index
        A1 = 0, B1 = 1, ..., G8 = 62, H8 = 63
        '''
        if self.myColor == self.PLAYING_COLOR_WHITE:
            x = self.lowerLeft[1] + (squareIndex % 8) * self.squareSize + int(self.squareSize/2)
            y = self.lowerLeft[0] - int(squareIndex / 8) * self.squareSize - int(self.squareSize/2)
            return (x,y)
        elif self.myColor == self.PLAYING_COLOR_BLACK:
            x = self.upperRight[1] - (squareIndex % 8) * self.squareSize - int(self.squareSize/2)
            y = self.upperRight[0] + int(squareIndex / 8) * self.squareSize + int(self.squareSize/2)
            return (x,y)

    def plotPieces(self):
        for p in self.parsedPieces:
            cv2.imshow(str(id(p)), p.astype('uint8'))
    
    def plotMasks(self):
        for p in self.parsedMasks:
            cv2.imshow(str(id(p)), p.astype('uint8')*255)

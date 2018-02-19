import argparse
import cv2
import numpy as np
import chess.uci
import chess.engine
from PIL import ImageGrab
from boardparser import boardParser
import pyautogui
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QErrorMessage, QMessageBox, QHBoxLayout
import threading
import time


class chessCheat(QWidget):

    def __init__(self, path):
        super().__init__()
        self.board = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        self.bp = boardParser()
        self.chessEngine = chess.uci.popen_engine(path)
        self.learnedPieces = False
        self.isPlaying = False
        self.waitForMoveThread = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("ChessCheat")
        self.resize(250, 0)
        hbox = QHBoxLayout()
        self.setLayout(hbox)
        buttonLearn = QPushButton('Learn')
        buttonLearn.clicked.connect(self.learn)
        hbox.addWidget(buttonLearn)
        buttonPlay = QPushButton('Play')
        buttonPlay.clicked.connect(self.play)
        hbox.addWidget(buttonPlay)
        self.show()
    
    def learn(self):
        inputImg = np.array(ImageGrab.grab())
    # try:
        self.bp.learnPieces(inputImg[...,::-1])
        gray = cv2.cvtColor(np.array(inputImg[...,::-1]),cv2.COLOR_RGB2GRAY)
        if self.bp.getPosition(gray) != 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR':
            QMessageBox.critical(self, "Error", "Chess board isn't in starting Position.")
            return
        self.learnedPieces = True
    # except:
    #     QMessageBox.critical(self, "Error", "No Chess Board found.")
    
    def play(self):
        if self.isPlaying == False and self.learnedPieces == True:
            self.isPlaying = True
            gray = cv2.cvtColor(np.array(ImageGrab.grab()),cv2.COLOR_BGR2GRAY)
            self.bp.findPlayingColour(gray)
            self.board = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
            if self.bp.myColor == boardParser.PLAYING_COLOR_WHITE:
                self.chessEngine.position(self.board)
                move = self.chessEngine.go(movetime=50)
                self.board.push(move.bestmove)
                self.movePiece(move.bestmove.from_square, move.bestmove.to_square)
            # self.waitForMoveThread = threading.Thread(target=self.waitForMove)
            # self.waitForMoveThread.daemon = True
            # self.waitForMoveThread.start()
            self.waitForMove()
            self.isPlaying = False
        
    def getAllPossibleNextBoards(self):
        possibleMoves = {}
        for move in self.board.legal_moves:
            self.board.push(move)
            possibleMoves[self.board.board_fen()] = move
            self.board.pop()
        return possibleMoves
    
    def waitForMove(self):
        next_call = time.time()
        changedStable = 0
        oldBoardPos = ''
        errorCounter = 0
        while True:
            if errorCounter > 3:
                QMessageBox.critical(self, "Error", "Detected illegal Position: " + currentBoardPos + "\nBoard Position: " + self.board.board_fen())
                return
            gray = cv2.cvtColor(np.array(ImageGrab.grab()),cv2.COLOR_BGR2GRAY)
            currentBoardPos = self.bp.getPosition(gray)
            if currentBoardPos != self.board.board_fen():
                if currentBoardPos == oldBoardPos:
                    if changedStable > 3:
                        changedStable = 0
                        legalMoves = self.getAllPossibleNextBoards()
                        if currentBoardPos in legalMoves:
                            errorCounter = 0
                            self.board.push(legalMoves[currentBoardPos])
                            self.chessEngine.position(self.board)
                            move = self.chessEngine.go(movetime=50)
                            self.board.push(move.bestmove)
                            self.movePiece(move.bestmove.from_square, move.bestmove.to_square)
                            if self.board.is_game_over(claim_draw=True):
                                QMessageBox.information(self, "Game Over", self.board.result()) 
                                return
                        else:
                            errorCounter +=1
                            changedStable = 0
                    else:
                        changedStable += 1
                else:
                    oldBoardPos = currentBoardPos
            next_call = next_call + 0.02
            s = next_call - time.time()
            if s < 0: s = 0
            time.sleep(s)
            
    def movePiece(self, fromSquare, toSquare):
        pyautogui.moveTo(self.bp.getSquarePosition(fromSquare))
        pyautogui.click(button='left')
        pyautogui.moveTo(self.bp.getSquarePosition(toSquare))
        pyautogui.click(button='left')
        pyautogui.moveTo((self.bp.lowerLeft[1]-10, self.bp.lowerLeft[0]+10)) # move Outside so Cursor doesnt interfere
        
    def on_show_message_box(self, id, severity, title, text):
        self.responses[str(id)] = getattr(QtGui.QMessageBox, str(severity))(self, title, text)
        
if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser(description = 'Chess Cheater')
    parser.add_argument('path', help='path to uci chess engine executable')
    args = parser.parse_args()
    app = QApplication(sys.argv)
    cc = chessCheat(args.path)
    sys.exit(app.exec_())

         

        

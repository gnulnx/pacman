#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pac-Man maze generator (Python renderer + bundled JS generator)

- Calls the original JS algorithm (unchanged) to produce the ASCII tile string,
  guaranteeing the same high-quality mazes as Shaun LeBron’s demo.
- Renders curved walls & pellets in Python via Pillow by porting map.js
  parseWalls/draw logic (including quadratic corner arcs via polyline sampling).

Usage:
  python pacman_mapgen.py              # single maze -> maze.png
  python pacman_mapgen.py --seed 42    # deterministic maze
  python pacman_mapgen.py --grid 4x3   # 4 cols × 3 rows collage -> grid.png
  python pacman_mapgen.py --out my.png # custom filename for single image
  python pacman_mapgen.py --tile 8     # change tile size (default 8)

Requires:
  - Python 3.9+
  - Pillow: pip install pillow
  - Node.js 16+: used to run the JS algorithm
"""

import argparse
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

# -------------------------------
# Embedded JS (Shaun LeBron’s algorithm)
# This is the *exact* logic from mapgen.js you pasted (genRandom + getTiles),
# plus tiny glue to print the tile string. No rendering, just the tiles.
# -------------------------------

JS_ALGO = r"""
// BEGIN: embedded pacman-mazegen (logic only) -------------------------------
var getRandomInt = function(min,max) {
    return Math.floor(Math.random() * (max-min+1)) + min;
};
var shuffle = function(list) {
    var len = list.length;
    for (var i=len-1; i>0; i--) {
        var j = getRandomInt(0,i);
        var t = list[i]; list[i] = list[j]; list[j] = t;
    }
};
var randomElement = function(list) {
    var len = list.length;
    if (len > 0) return list[getRandomInt(0,len-1)];
};

var UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3;

var cells = [];
var tallRows = [];
var narrowCols = [];

var rows = 9;
var cols = 5;

function reset(seed) {
    if (seed !== undefined) {                // simple LCG for deterministic mode
      let s = Number(seed)>>>0;
      Math.random = function(){
        s = (1664525*s + 1013904223) >>> 0;
        return (s >>> 8) / (1<<24);
      }
    }

    cells = [];
    for (var i=0; i<rows*cols; i++) {
        cells[i] = { x:i%cols, y:Math.floor(i/cols), filled:false,
            connect:[false,false,false,false], next:[], no:undefined, group:undefined };
    }
    for (var i=0; i<rows*cols; i++) {
        var c = cells[i];
        if (c.x>0) c.next[LEFT]=cells[i-1];
        if (c.x<cols-1) c.next[RIGHT]=cells[i+1];
        if (c.y>0) c.next[UP]=cells[i-cols];
        if (c.y<rows-1) c.next[DOWN]=cells[i+cols];
    }
    var i=3*cols, c=cells[i];
    c.filled=true; c.connect[LEFT]=c.connect[RIGHT]=c.connect[DOWN]=true;
    i++; c=cells[i]; c.filled=true; c.connect[LEFT]=c.connect[DOWN]=true;
    i+=cols-1; c=cells[i]; c.filled=true; c.connect[LEFT]=c.connect[UP]=c.connect[RIGHT]=true;
    i++; c=cells[i]; c.filled=true; c.connect[UP]=c.connect[LEFT]=true;
}

function genRandom(seed){
  reset(seed);

  function getLeftMostEmptyCells(){
    var left=[];
    for (var x=0; x<cols; x++){
      for (var y=0; y<rows; y++){
        var c=cells[x+y*cols];
        if(!c.filled) left.push(c);
      }
      if(left.length>0) break;
    }
    return left;
  }
  function isOpenCell(cell,i,prevDir,size){
    if ((cell.y==6 && cell.x==0 && i==DOWN) || (cell.y==7 && cell.x==0 && i==UP)) return false;
    if (size==2 && (i==prevDir || (i+2)%4==prevDir)) return false;
    if (cell.next[i] && !cell.next[i].filled){
      if (cell.next[i].next[LEFT] && !cell.next[i].next[LEFT].filled) {}
      else return true;
    }
    return false;
  }
  function getOpenCells(cell,prevDir,size){
    var open=[], n=0;
    for (var i=0;i<4;i++){ if (isOpenCell(cell,i,prevDir,size)){ open.push(i); n++; } }
    return {openCells:open, numOpenCells:n};
  }
  function connectCell(cell,dir){
    cell.connect[dir]=true;
    cell.next[dir].connect[(dir+2)%4]=true;
    if (cell.x==0 && dir==RIGHT) cell.connect[LEFT]=true;
  }

  function gen(){
    var cell,newCell,firstCell,openCells,numOpenCells,dir,i;
    var numFilled=0, numGroups, size;
    var probStop=[0,0,0.10,0.5,0.75,1];
    var singleCount={}; singleCount[0]=singleCount[rows-1]=0;
    var probTopBotSingle=0.35;
    var longPieces=0, maxLongPieces=1, pSize2=1, pSize34=0.5;
    function fillCell(c){ c.filled=true; c.no=numFilled++; c.group=numGroups; }

    for (numGroups=0;;numGroups++){
      openCells=getLeftMostEmptyCells(); numOpenCells=openCells.length;
      if (numOpenCells==0) break;
      firstCell=cell=openCells[getRandomInt(0,numOpenCells-1)];
      fillCell(cell);

      if (cell.x<cols-1 && (cell.y in singleCount) && Math.random()<=probTopBotSingle){
        if (singleCount[cell.y]==0){ cell.connect[cell.y==0?UP:DOWN]=true; singleCount[cell.y]++; continue; }
      }
      size=1;
      if (cell.x==cols-1){ cell.connect[RIGHT]=true; cell.isRaiseHeightCandidate=true; }
      else {
        while (size<5){
          var stop=false;

          if (size==2){
            var c=firstCell;
            if (c.x>0 && c.connect[RIGHT] && c.next[RIGHT] && c.next[RIGHT].next[RIGHT]){
              if (longPieces<maxLongPieces && Math.random()<=pSize2){
                c=c.next[RIGHT].next[RIGHT];
                var dirs={};
                if (isOpenCell(c,UP)) dirs[UP]=true;
                if (isOpenCell(c,DOWN)) dirs[DOWN]=true;
                if (dirs[UP] && dirs[DOWN]) i=[UP,DOWN][getRandomInt(0,1)];
                else if (dirs[UP]) i=UP;
                else if (dirs[DOWN]) i=DOWN;
                else i=undefined;
                if (i!=undefined){
                  connectCell(c,LEFT); fillCell(c);
                  connectCell(c,i);    fillCell(c.next[i]);
                  longPieces++; size+=2; stop=true;
                }
              }
            }
          }

          if (!stop){
            var res=getOpenCells(cell,dir,size);
            openCells=res.openCells; numOpenCells=res.numOpenCells;
            if (numOpenCells==0 && size==2){
              cell=newCell;
              res=getOpenCells(cell,dir,size);
              openCells=res.openCells; numOpenCells=res.numOpenCells;
            }
            if (numOpenCells==0){ stop=true; }
            else{
              dir=openCells[getRandomInt(0,numOpenCells-1)];
              newCell=cell.next[dir];
              connectCell(cell,dir); fillCell(newCell); size++;
              if (firstCell.x==0 && size==3) stop=true;
              if (Math.random()<=probStop[size]) stop=true;
            }
          }

          if (stop){
            if (size==2){
              var c=firstCell;
              if (c.x==cols-1){
                if (c.connect[UP]) c=c.next[UP];
                c.connect[RIGHT]=c.next[DOWN].connect[RIGHT]=true;
              }
            } else if (size==3 || size==4){
              if (longPieces<maxLongPieces && firstCell.x>0 && Math.random()<=pSize34){
                var dirs=[], dl=0;
                for (i=0;i<4;i++){
                  if (cell.connect[i] && isOpenCell(cell.next[i],i)){ dirs.push(i); dl++; }
                }
                if (dl>0){
                  i=dirs[getRandomInt(0,dl-1)];
                  c=cell.next[i]; connectCell(c,i); fillCell(c.next[i]); longPieces++;
                }
              }
            }
            break;
          }
        }
      }
    }
    setResizeCandidates();
  }

  function setResizeCandidates(){
    for (var i=0;i<rows*cols;i++){
      var c=cells[i], x=i%cols, y=Math.floor(i/cols), q=c.connect;

      if ((c.x==0 || !q[LEFT]) && (c.x==cols-1 || !q[RIGHT]) && q[UP]!=q[DOWN]) {
        c.isRaiseHeightCandidate=true;
      }
      var c2=c.next[RIGHT]; if (c2){
        var q2=c2.connect;
        if (((c.x==0 || !q[LEFT]) && !q[UP] && !q[DOWN]) && ((c2.x==cols-1 || !q2[RIGHT]) && !q2[UP] && !q2[DOWN])){
          c.isRaiseHeightCandidate=c2.isRaiseHeightCandidate=true;
        }
      }
      if (c.x==cols-1 && q[RIGHT]) c.isShrinkWidthCandidate=true;
      if ((c.y==0||!q[UP]) && (c.y==rows-1||!q[DOWN]) && q[LEFT]!=q[RIGHT]) c.isShrinkWidthCandidate=true;
    }
  }

  function cellIsCrossCenter(c){ return c.connect[UP]&&c.connect[RIGHT]&&c.connect[DOWN]&&c.connect[LEFT]; }

  function chooseNarrowCols(){
    function canShrinkWidth(x,y){
      if (y==rows-1) return true;
      var x0,c,c2;
      for(x0=x;x0<cols;x0++){
        c=cells[x0+y*cols]; c2=c.next[DOWN];
        if((!c.connect[RIGHT]||cellIsCrossCenter(c)) && (!c2.connect[RIGHT]||cellIsCrossCenter(c2))) break;
      }
      var candidates=[], n=0;
      for(;c2;c2=c2.next[LEFT]){
        if (c2.isShrinkWidthCandidate){ candidates.push(c2); n++; }
        if((!c2.connect[LEFT]||cellIsCrossCenter(c2)) && (!c2.next[UP].connect[LEFT]||cellIsCrossCenter(c2.next[UP]))) break;
      }
      shuffle(candidates);
      for (var i=0;i<n;i++){
        c2=candidates[i];
        if (canShrinkWidth(c2.x,c2.y)){ c2.shrinkWidth=true; narrowCols[c2.y]=c2.x; return true; }
      }
      return false;
    }
    for (var x=cols-1;x>=0;x--){
      var c=cells[x];
      if (c.isShrinkWidthCandidate && canShrinkWidth(x,0)){ c.shrinkWidth=true; narrowCols[c.y]=c.x; return true; }
    }
    return false;
  }

  function chooseTallRows(){
    function canRaiseHeight(x,y){
      if (x==cols-1) return true;
      var y0,c,c2;
      for(y0=y;y0>=0;y0--){
        c=cells[x+y0*cols]; c2=c.next[RIGHT];
        if((!c.connect[UP]||cellIsCrossCenter(c)) && (!c2.connect[UP]||cellIsCrossCenter(c2))) break;
      }
      var candidates=[], n=0;
      for(;c2;c2=c2.next[DOWN]){
        if(c2.isRaiseHeightCandidate){ candidates.push(c2); n++; }
        if((!c2.connect[DOWN]||cellIsCrossCenter(c2)) && (!c2.next[LEFT].connect[DOWN]||cellIsCrossCenter(c2.next[LEFT]))) break;
      }
      shuffle(candidates);
      for (var i=0;i<n;i++){
        c2=candidates[i];
        if (canRaiseHeight(c2.x,c2.y)){ c2.raiseHeight=true; tallRows[c2.x]=c2.y; return true; }
      }
      return false;
    }
    for (var y=0;y<3;y++){
      var c=cells[y*cols];
      if (c.isRaiseHeightCandidate && canRaiseHeight(0,y)){ c.raiseHeight=true; tallRows[c.x]=c.y; return true; }
    }
    return false;
  }

  function isDesirable(){
    var c=cells[4]; if (c.connect[UP] || c.connect[RIGHT]) return false;
    c=cells[rows*cols-1]; if (c.connect[DOWN]||c.connect[RIGHT]) return false;

    function isH(x,y){ var q1=cells[x+y*cols].connect, q2=cells[x+1+y*cols].connect;
      return !q1[UP]&&!q1[DOWN]&&(x==0||!q1[LEFT])&&q1[RIGHT] && !q2[UP]&&!q2[DOWN]&&q2[LEFT]&&!q2[RIGHT];
    }
    function isV(x,y){ var q1=cells[x+y*cols].connect, q2=cells[x+(y+1)*cols].connect;
      if (x==cols-1) return !q1[LEFT]&&!q1[UP]&&!q1[DOWN] && !q2[LEFT]&&!q2[UP]&&!q2[DOWN];
      return !q1[LEFT]&&!q1[RIGHT]&&!q1[UP]&&q1[DOWN] && !q2[LEFT]&&!q2[RIGHT]&&q2[UP]&&!q2[DOWN];
    }
    for (var y=0;y<rows-1;y++){
      for (var x=0;x<cols-1;x++){
        if ( (isH(x,y)&&isH(x,y+1)) || (isV(x,y)&&isV(x+1,y)) ){
          if (x==0) return false;
          cells[x+y*cols].connect[DOWN]=true; cells[x+y*cols].connect[RIGHT]=true; var g=cells[x+y*cols].group;
          cells[x+1+y*cols].connect[DOWN]=true; cells[x+1+y*cols].connect[LEFT]=true; cells[x+1+y*cols].group=g;
          cells[x+(y+1)*cols].connect[UP]=true; cells[x+(y+1)*cols].connect[RIGHT]=true; cells[x+(y+1)*cols].group=g;
          cells[x+1+(y+1)*cols].connect[UP]=true; cells[x+1+(y+1)*cols].connect[LEFT]=true; cells[x+1+(y+1)*cols].group=g;
        }
      }
    }
    if (!chooseTallRows()) return false;
    if (!chooseNarrowCols()) return false;
    return true;
  }

  function setUpScale(){
    for (var i=0;i<rows*cols;i++){
      var c=cells[i];
      c.final_x=c.x*3; if (narrowCols[c.y] < c.x) c.final_x--;
      c.final_y=c.y*3; if (tallRows[c.x]   < c.y) c.final_y++;
      c.final_w=c.shrinkWidth?2:3;
      c.final_h=c.raiseHeight?4:3;
    }
  }

  function createTunnels(){
    var single=[], topS=[], botS=[];
    var voids=[], topV=[], botV=[];
    var edge=[],  topE=[], botE=[];
    var doubles=[];
    var numT=0;

    for (var y=0;y<rows;y++){
      var c=cells[cols-1+y*cols];
      if (c.connect[UP]) continue;
      if (c.y>1 && c.y<rows-2){ c.isEdgeTunnelCandidate=true; edge.push(c); if(c.y<=2)topE.push(c); else if(c.y>=5)botE.push(c); }
      var upDead=(!c.next[UP]||c.next[UP].connect[RIGHT]);
      var dnDead=(!c.next[DOWN]||c.next[DOWN].connect[RIGHT]);
      if (c.connect[RIGHT]){
        if (upDead){ c.isVoidTunnelCandidate=true; voids.push(c); if(c.y<=2)topV.push(c); else if(c.y>=6)botV.push(c); }
      } else {
        if (c.connect[DOWN]) continue;
        if (upDead != dnDead){
          if(!c.raiseHeight && y<rows-1 && !c.next[LEFT].connect[LEFT]){
            single.push(c); c.isSingleDeadEndCandidate=true; c.singleDeadEndDir=upDead?UP:DOWN;
            var off = upDead?1:0;
            if (c.y<=1+off) topS.push(c); else if (c.y>=5+off) botS.push(c);
          }
        } else if (upDead && dnDead){
          if (y>0 && y<rows-1){
            if (c.next[LEFT].connect[UP] && c.next[LEFT].connect[DOWN]){
              c.isDoubleDeadEndCandidate=true;
              if (c.y>=2 && c.y<=5) doubles.push(c);
            }
          }
        }
      }
    }

    function pickSingle(c){ c.connect[RIGHT]=true; if (c.singleDeadEndDir==UP) c.topTunnel=true; else c.next[DOWN].topTunnel=true; }

    var c;
    var want = Math.random()<=0.45 ? 2 : 1;
    if (want==1){
      if      (c=randomElement(voids))  c.topTunnel=true;
      else if (c=randomElement(single)) pickSingle(c);
      else if (c=randomElement(edge))   c.topTunnel=true;
      else return false;
    } else {
      if (c=randomElement(doubles)){
        c.connect[RIGHT]=true; c.topTunnel=true; c.next[DOWN].topTunnel=true;
      } else {
        var made=0;
        if      (c=randomElement(topV)) { c.topTunnel=true; made=1; }
        else if (c=randomElement(topS)) { pickSingle(c);   made=1; }
        else if (c=randomElement(topE)) { c.topTunnel=true; made=1; }
        if      (c=randomElement(botV)) { c.topTunnel=true; }
        else if (c=randomElement(botS)) { pickSingle(c);   }
        else if (c=randomElement(botE)) { c.topTunnel=true; }
        else if (!made) return false;
      }
    }

    // forbid straight-through corridors
    for (var y=0;y<rows;y++){
      c=cells[cols-1+y*cols];
      if (c.topTunnel){
        var exit=true, topy=c.final_y;
        while(c.next[LEFT]){
          c=c.next[LEFT];
          if(!c.connect[UP] && c.final_y==topy) continue;
          else { exit=false; break; }
        }
        if (exit) return false;
      }
    }

    // clear unused void tunnels
    for (var k=0;k<voids.length;k++){
      c=voids[k];
      if(!c.topTunnel){
        var og=c.group, ng=c.next[UP].group;
        for (var i=0;i<rows*cols;i++){ var u=cells[i]; if(u.group==og) u.group=ng; }
        c.connect[UP]=true; c.next[UP].connect[DOWN]=true;
      }
    }
    return true;
  }

  function joinWalls(){
    var x,y,c,c2;
    for (x=0;x<cols;x++){
      c=cells[x];
      if(!c.connect[LEFT]&&!c.connect[RIGHT]&&!c.connect[UP] && (!c.connect[DOWN]||!c.next[DOWN].connect[DOWN])){
        if((!c.next[LEFT]||!c.next[LEFT].connect[UP]) && (c.next[RIGHT] && !c.next[RIGHT].connect[UP])){
          if(!(c.next[DOWN]&&c.next[DOWN].connect[RIGHT]&&c.next[DOWN].next[RIGHT].connect[RIGHT])){
            c.isJoinCandidate=true; if(Math.random()<=0.25) c.connect[UP]=true;
          }
        }
      }
    }
    for (x=0;x<cols;x++){
      c=cells[x+(rows-1)*cols];
      if(!c.connect[LEFT]&&!c.connect[RIGHT]&&!c.connect[DOWN] && (!c.connect[UP]||!c.next[UP].connect[UP])){
        if((!c.next[LEFT]||!c.next[LEFT].connect[DOWN]) && (c.next[RIGHT] && !c.next[RIGHT].connect[DOWN])){
          if(!(c.next[UP]&&c.next[UP].connect[RIGHT]&&c.next[UP].next[RIGHT].connect[RIGHT])){
            c.isJoinCandidate=true; if(Math.random()<=0.25) c.connect[DOWN]=true;
          }
        }
      }
    }
    for (y=1;y<rows-1;y++){
      c=cells[cols-1+y*cols];
      if(c.raiseHeight) continue;
      if(!c.connect[RIGHT]&&!c.connect[UP]&&!c.connect[DOWN] && !c.next[UP].connect[RIGHT]&&!c.next[DOWN].connect[RIGHT]){
        if(c.connect[LEFT]){
          c2=c.next[LEFT];
          if(!c2.connect[UP]&&!c2.connect[DOWN]&&!c2.connect[LEFT]){
            c.isJoinCandidate=true; if(Math.random()<=0.5) c.connect[RIGHT]=true;
          }
        }
      }
    }
  }

  var tries=0;
  while(tries < 1000){
    tries++;
    gen();
    if(!isDesirable()) continue;
    setUpScale();
    joinWalls();
    if(!createTunnels()) continue;
    break;
  }
}

function getTiles(){
  var tiles=[];
  var tileCells=[];
  var subrows=rows*3+1+3;
  var subcols=cols*3-1+2;
  var midcols=subcols-2;
  var fullcols=(subcols-2)*2;

  function setTile(x,y,v){
    if(x<0||x>subcols-1||y<0||y>subrows-1) return;
    x -= 2;
    tiles[midcols+x+y*fullcols]=v;
    tiles[midcols-1-x+y*fullcols]=v;
  }
  function getTile(x,y){
    if(x<0||x>subcols-1||y<0||y>subrows-1) return undefined;
    x -= 2;
    return tiles[midcols+x+y*fullcols];
  }
  function setTileCell(x,y,cell){
    if(x<0||x>subcols-1||y<0||y>subrows-1) return;
    x -= 2;
    tileCells[x+y*subcols]=cell;
  }
  function getTileCell(x,y){
    if(x<0||x>subcols-1||y<0||y>subrows-1) return undefined;
    x -= 2;
    return tileCells[x+y*subcols];
  }

  for (var i=0;i<subrows*fullcols;i++) tiles.push('_');
  for (var i=0;i<subrows*subcols;i++) tileCells.push(undefined);

  for (var i=0;i<rows*cols;i++){
    var c=cells[i];
    for (var x0=0;x0<c.final_w;x0++){
      for (var y0=0;y0<c.final_h;y0++){
        setTileCell(c.final_x+x0,c.final_y+1+y0,c);
      }
    }
  }

  for (var y=0;y<subrows;y++){
    for (var x=0;x<subcols;x++){
      var c=getTileCell(x,y), cl=getTileCell(x-1,y), cu=getTileCell(x,y-1);
      if (c){
        if (cl && c.group != cl.group || cu && c.group != cu.group || !cu && !c.connect[UP]) {
          setTile(x,y,'.');
        }
      } else {
        if (cl && (!cl.connect[RIGHT] || getTile(x-1,y)=='.') ||
            cu && (!cu.connect[DOWN]  || getTile(x,y-1)=='.')) {
          setTile(x,y,'.');
        }
      }
      if (getTile(x-1,y)=='.' && getTile(x,y-1)=='.' && getTile(x-1,y-1)=='_'){
        setTile(x,y,'.');
      }
    }
  }

  for (var c=cells[cols-1]; c; c=c.next[DOWN]){
    if (c.topTunnel){
      var y=c.final_y+1;
      setTile(subcols-1,y,'.'); setTile(subcols-2,y,'.');
    }
  }

  for (var y=0;y<subrows;y++){
    for (var x=0;x<subcols;x++){
      if (getTile(x,y)!='.' && (getTile(x-1,y)=='.'||getTile(x,y-1)=='.'||getTile(x+1,y)=='.'||getTile(x,y+1)=='.'||
          getTile(x-1,y-1)=='.'||getTile(x+1,y-1)=='.'||getTile(x+1,y+1)=='.'||getTile(x-1,y+1)=='.')){
        setTile(x,y,'|');
      }
    }
  }

  setTile(2,12,'-');

  function getTopRange(){
    var miny, maxy=subrows/2, x=subcols-2;
    for (var y=2;y<maxy;y++){ if(getTile(x,y)=='.'&&getTile(x,y+1)=='.'){ miny=y+1; break; } }
    maxy=Math.min(maxy,miny+7);
    for (var y=miny+1;y<maxy;y++){ if(getTile(x-1,y)=='.'){ maxy=y-1; break; } }
    return {miny:miny, maxy:maxy};
  }
  function getBotRange(){
    var miny=subrows/2, maxy, x=subcols-2;
    for (var y=subrows-3;y>=miny;y--){ if(getTile(x,y)=='.'&&getTile(x,y+1)=='.'){ maxy=y; break; } }
    miny=Math.max(miny,maxy-7);
    for (var y=maxy-1;y>miny;y--){ if(getTile(x-1,y)=='.'){ miny=y+1; break; } }
    return {miny:miny, maxy:maxy};
  }
  var x=subcols-2, range, y;
  if (range=getTopRange()){ y=getRandomInt(range.miny,range.maxy); setTile(x,y,'o'); }
  if (range=getBotRange()){ y=getRandomInt(range.miny,range.maxy); setTile(x,y,'o'); }

  function eraseUntil(x,y){
    while(true){
      var adj=[];
      if(getTile(x-1,y)=='.') adj.push({x:x-1,y:y});
      if(getTile(x+1,y)=='.') adj.push({x:x+1,y:y});
      if(getTile(x,y-1)=='.') adj.push({x:x,y:y-1});
      if(getTile(x,y+1)=='.') adj.push({x:x,y:y+1});
      if(adj.length==1){ setTile(x,y,' '); x=adj[0].x; y=adj[0].y; }
      else break;
    }
  }
  x=subcols-1;
  for (var y=0;y<subrows;y++){ if(getTile(x,y)=='.') eraseUntil(x,y); }

  setTile(1,subrows-8,' ');

  for (var i=0;i<7;i++){
    var y=subrows-14; setTile(i,y,' '); var j=1;
    while(getTile(i,y+j)=='.' && getTile(i-1,y+j)=='|' && getTile(i+1,y+j)=='|'){ setTile(i,y+j,' '); j++; }
    y=subrows-20; setTile(i,y,' '); j=1;
    while(getTile(i,y-j)=='.' && getTile(i-1,y-j)=='|' && getTile(i+1,y-j)=='|'){ setTile(i,y-j,' '); j++; }
  }
  for (var i=0;i<7;i++){
    var x=6, y=subrows-14-i; setTile(x,y,' '); var j=1;
    while(getTile(x+j,y)=='.' && getTile(x+j,y-1)=='|' && getTile(x+j,y+1)=='|'){ setTile(x+j,y,' '); j++; }
  }

  return "____________________________".repeat(3) + tiles.join("") + "____________________________".repeat(2);
}

// Entrypoint: generate and print tiles (optionally seeded)
(function(){
  var seed = process.env.PACMAN_SEED;
  genRandom(seed);
  var t = getTiles();
  process.stdout.write(t);
})();
 // END: embedded pacman-mazegen ---------------------------------------------
"""

# -------------------------------
# Python port of map.js wall pathing + renderer
# -------------------------------

TILE_SIZE_DEFAULT = 8


def rgb(hexstr: str) -> Tuple[int, int, int]:
    hexstr = hexstr.lstrip("#")
    return tuple(int(hexstr[i : i + 2], 16) for i in (0, 2, 4))


def qbezier(p0, p1, c, steps=16) -> List[Tuple[float, float]]:
    """Sample a quadratic Bezier from p0 -> p1 with control c."""
    pts = []
    for k in range(steps + 1):
        t = k / steps
        x = (1 - t) * (1 - t) * p0[0] + 2 * (1 - t) * t * c[0] + t * t * p1[0]
        y = (1 - t) * (1 - t) * p0[1] + 2 * (1 - t) * t * c[1] + t * t * p1[1]
        pts.append((x, y))
    return pts


@dataclass
class MapPy:
    numCols: int
    numRows: int
    tiles: str
    tileSize: int = TILE_SIZE_DEFAULT
    wallFillColor: Tuple[int, int, int] = (0, 51, 255)
    wallStrokeColor: Tuple[int, int, int] = (255, 255, 255)
    pelletColor: Tuple[int, int, int] = (255, 184, 174)

    def __post_init__(self):
        self.widthPixels = self.numCols * self.tileSize
        self.heightPixels = self.numRows * self.tileSize
        self.resetCurrent()
        self.parseWalls()

    def resetCurrent(self):
        self.currentTiles = list(self.tiles)

    def posToIndex(self, x: int, y: int) -> Optional[int]:
        if 0 <= x < self.numCols and 0 <= y < self.numRows:
            return x + y * self.numCols
        return None

    def getTile(self, x: int, y: int) -> Optional[str]:
        idx = self.posToIndex(x, y)
        if idx is not None:
            return self.currentTiles[idx]
        # tunnel extension logic (copy of map.js)
        if (
            x == -1
            and self.getTile(x + 1, y) == "|"
            and (self.isFloorTile(x + 1, y + 1) or self.isFloorTile(x + 1, y - 1))
        ) or (
            x == self.numCols
            and self.getTile(x - 1, y) == "|"
            and (self.isFloorTile(x - 1, y + 1) or self.isFloorTile(x - 1, y - 1))
        ):
            return "|"
        if (x == -1 and self.isFloorTile(x + 1, y)) or (x == self.numCols and self.isFloorTile(x - 1, y)):
            return " "
        return None

    @staticmethod
    def isFloorTileChar(ch: str) -> bool:
        return ch in (" ", ".", "o")

    def isFloorTile(self, x: int, y: int) -> bool:
        t = self.getTile(x, y)
        return t is not None and self.isFloorTileChar(t)

    # --- parseWalls port (build curved polygon paths) ---
    def parseWalls(self):
        DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT = 0, 1, 2, 3

        def setDirFromEnum(dir_dict, de):
            if de == DIR_UP:
                dir_dict["x"], dir_dict["y"] = (0, -1)
            elif de == DIR_RIGHT:
                dir_dict["x"], dir_dict["y"] = (1, 0)
            elif de == DIR_DOWN:
                dir_dict["x"], dir_dict["y"] = (0, 1)
            else:
                dir_dict["x"], dir_dict["y"] = (-1, 0)

        self.paths = []
        visited: Dict[int, bool] = {}

        def toIndex(x, y):
            if -2 <= x < self.numCols + 2 and 0 <= y < self.numRows:
                return (x + 2) + y * (self.numCols + 4)

        edges: Dict[int, bool] = {}
        i = 0
        for y in range(self.numRows):
            for x in range(-2, self.numCols + 2):
                t = self.getTile(x, y)
                if t == "|":
                    neigh = [
                        self.getTile(x - 1, y),
                        self.getTile(x + 1, y),
                        self.getTile(x, y - 1),
                        self.getTile(x, y + 1),
                        self.getTile(x - 1, y - 1),
                        self.getTile(x - 1, y + 1),
                        self.getTile(x + 1, y - 1),
                        self.getTile(x + 1, y + 1),
                    ]
                    if any(nt != "|" for nt in neigh):
                        edges[i] = True
                i += 1

        def getStartPoint(tx, ty, dirEnum, pad_state):
            d = {"x": 0, "y": 0}
            setDirFromEnum(d, dirEnum)
            if toIndex(tx + d["y"], ty - d["x"]) not in edges:
                pad_state["pad"] = 5 if self.isFloorTile(tx + d["y"], ty - d["x"]) else 0
            px = -self.tileSize / 2 + pad_state["pad"]
            py = self.tileSize / 2
            a = dirEnum * math.pi / 2
            c, s = math.cos(a), math.sin(a)
            x = (px * c - py * s) + (tx + 0.5) * self.tileSize
            y = (px * s + py * c) + (ty + 0.5) * self.tileSize
            return {"x": x, "y": y}

        def makePath(sx, sy):
            d = {"x": 0, "y": 0}
            if toIndex(sx + 1, sy) in edges:
                dirEnum = DIR_RIGHT
            elif toIndex(sx, sy + 1) in edges:
                dirEnum = DIR_DOWN
            else:
                raise RuntimeError(f"1x1 tile at {sx},{sy}")
            setDirFromEnum(d, dirEnum)
            tx, ty = sx + d["x"], sy + d["y"]
            init_tx, init_ty, init_de = tx, ty, dirEnum
            path = []
            pad_state = {"pad": 0}
            turn = False
            turnAround = False

            while True:
                visited[toIndex(tx, ty)] = True
                pt = getStartPoint(tx, ty, dirEnum, pad_state)

                if turn:
                    last = path[-1]
                    if d["x"] == 0:
                        pt["cx"] = pt["x"]
                        pt["cy"] = last["y"]
                    else:
                        pt["cx"] = last["x"]
                        pt["cy"] = pt["y"]

                turn = False
                turnAround = False
                if toIndex(tx + d["y"], ty - d["x"]) in edges:
                    dirEnum = (dirEnum + 3) % 4
                    turn = True
                elif toIndex(tx + d["x"], ty + d["y"]) in edges:
                    pass
                elif toIndex(tx - d["y"], ty + d["x"]) in edges:
                    dirEnum = (dirEnum + 1) % 4
                    turn = True
                else:
                    dirEnum = (dirEnum + 2) % 4
                    turnAround = True
                setDirFromEnum(d, dirEnum)

                path.append(pt)
                if turnAround:
                    back = getStartPoint(tx - d["x"], ty - d["y"], (dirEnum + 2) % 4, pad_state)
                    path.append(back)
                    again = getStartPoint(tx, ty, dirEnum, pad_state)
                    path.append(again)

                tx += d["x"]
                ty += d["y"]
                if tx == init_tx and ty == init_ty and dirEnum == init_de:
                    self.paths.append(path)
                    break

        i = 0
        for y in range(self.numRows):
            for x in range(-2, self.numCols + 2):
                idx = i
                i += 1
                if (idx in edges) and (idx not in visited):
                    visited[idx] = True
                    makePath(x, y)

    def draw(self, out_path: str, print_mode: bool = False):
        ts = self.tileSize
        img = Image.new("RGB", (self.widthPixels, self.heightPixels), (0, 0, 0) if not print_mode else (255, 255, 255))
        draw = ImageDraw.Draw(img, "RGBA")

        # fill wall shapes (curved)
        fill = self.wallFillColor if not print_mode else (51, 51, 51)
        stroke = self.wallStrokeColor if not print_mode else (51, 51, 51)

        for path in self.paths:
            if not path:
                continue
            poly = []
            for k in range(1, len(path)):
                a = path[k - 1]
                b = path[k]
                if "cx" in b and "cy" in b:
                    seg = qbezier((a["x"], a["y"]), (b["x"], b["y"]), (b["cx"], b["cy"]), steps=16)
                    if poly:
                        seg = seg[1:]
                    poly.extend(seg)
                else:
                    poly.append((b["x"], b["y"]))
            # close with a final quadratic to the start as in map.js
            a = path[-1]
            b = path[0]
            seg = qbezier((a["x"], a["y"]), (b["x"], b["y"]), (a["x"], b["y"]), steps=16)
            if poly:
                seg = seg[1:]
            poly.extend(seg)

            draw.polygon(poly, fill=fill)
            draw.line(poly + [poly[0]], fill=stroke, width=1)

        # pellets
        pellet = (187, 187, 187) if print_mode else self.pelletColor
        pellet_size = ts if print_mode else 2
        for y in range(self.numRows):
            for x in range(self.numCols):
                t = self.getTile(x, y)
                if t in (".", "o", " "):
                    cx = x * ts + ts / 2
                    cy = y * ts + ts / 2
                    r = 3 if (t == "o" and not print_mode) else pellet_size / 2
                    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=pellet)

        # grid (subtle)
        grid = (0, 0, 0, 77) if print_mode else (255, 255, 255, 77)
        for y in range(self.numRows + 1):
            draw.line([(0, y * ts), (self.widthPixels, y * ts)], fill=grid, width=1)
        for x in range(self.numCols + 1):
            draw.line([(x * ts, 0), (x * ts, self.heightPixels)], fill=grid, width=1)

        img.save(out_path)
        return img


# -------------------------------
# JS invocation util
# -------------------------------


def generate_tiles_with_js(seed: Optional[int] = None) -> str:
    if not shutil.which("node"):
        print("ERROR: Node.js is required to run the embedded generator. Please install Node 16+.", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as td:
        js_path = os.path.join(td, "gen.js")
        with open(js_path, "w", encoding="utf-8") as f:
            f.write(JS_ALGO)
        env = os.environ.copy()
        if seed is not None:
            env["PACMAN_SEED"] = str(seed)
        try:
            out = subprocess.check_output(["node", js_path], env=env)
        except subprocess.CalledProcessError as e:
            print("JS generator failed:", e, file=sys.stderr)
            sys.exit(1)
    tiles = out.decode("utf-8")
    return tiles


# -------------------------------
# Collage
# -------------------------------


def render_single(seed: Optional[int], tile_size: int, out_path: str):
    tiles = generate_tiles_with_js(seed)
    # The algorithm emits a 28×36 ASCII field (incl. padding rows)
    COLS, ROWS = 28, 36
    m = MapPy(
        COLS,
        ROWS,
        tiles,
        tileSize=tile_size,
        wallFillColor=rgb("#00d0ff"),
        wallStrokeColor=rgb("#8cf0ff"),
        pelletColor=rgb("#ffb8ae"),
    )
    m.draw(out_path)
    print(f"✅ Maze image saved to {out_path}")


def render_grid(grid_spec: str, seed: Optional[int], tile_size: int, out_path: str):
    cols, rows = map(int, grid_spec.lower().split("x"))
    COLS, ROWS = 28, 36
    cell_w, cell_h = COLS * tile_size, ROWS * tile_size
    gap = tile_size  # spacing between mazes
    W = cols * cell_w + (cols - 1) * gap
    H = rows * cell_h + (rows - 1) * gap
    canvas = Image.new("RGB", (W, H), (20, 20, 24))

    rng = random.Random(seed)
    for r in range(rows):
        for c in range(cols):
            s = rng.randrange(1 << 30) if seed is not None else None
            tiles = generate_tiles_with_js(s)
            # fun palette per cell
            hue = rng.random()
            wall = tuple(int(v) for v in (0, 180 + 60 * hue, 255 * hue + 80))
            stroke = (255, 255, 255)
            m = MapPy(
                COLS,
                ROWS,
                tiles,
                tileSize=tile_size,
                wallFillColor=wall,
                wallStrokeColor=stroke,
                pelletColor=(255, 184, 174),
            )
            img = m.draw(out_path=None, print_mode=False)
            x0 = c * (cell_w + gap)
            y0 = r * (cell_h + gap)
            canvas.paste(img, (x0, y0))

    canvas.save(out_path)
    print(f"✅ Grid saved to {out_path}")


# -------------------------------
# CLI
# -------------------------------


def main():
    p = argparse.ArgumentParser(description="Pac-Man maze generator (Python renderer + JS algorithm)")
    p.add_argument("--seed", type=int, help="deterministic seed for the maze")
    p.add_argument("--tile", type=int, default=TILE_SIZE_DEFAULT, help="tile size in pixels (default 8)")
    p.add_argument("--grid", type=str, help="e.g. 4x3 to render a collage instead of a single maze")
    p.add_argument("--out", type=str, help="output filename (default: maze.png or grid.png)")
    args = p.parse_args()

    if args.grid:
        out = args.out or "grid.png"
        render_grid(args.grid, args.seed, args.tile, out)
    else:
        out = args.out or "maze.png"
        render_single(args.seed, args.tile, out)


if __name__ == "__main__":
    main()

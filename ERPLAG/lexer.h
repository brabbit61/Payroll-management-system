/*
 *	COMPILER PROJECT- ERPLAG COMPILER
 *	CSE - D 2019
 *	Sahil Garg, Jenit Jain, Amrit Goyal
 */

#ifndef lexer
#define lexer
#include "lexerDef.h"

#endif

//List of functions for Hashing for Keywords
hashTable initializeHashTable(int l);
int hashValue(char *key);
store findKeyword(hashTable T, char *searchKey);
void addKeyword(hashTable T, int c, char *key);

//List of functions for Lexemes/Tokens
tokenInfo* initToken();
void addDollar();
void addSym(int n);
void addID(char *id);
void addRNum(float num, int rnum_no);
void addNum(int num, int num_no);
void DFA(hashTable table, char *buf, int pr);
char* getStream();
tokenInfo* getFirstToken();
tokenInfo* getNextToken();
void removeComments(char *testcaseFile, char *cleanFile);
int populateLexemeTable(char *opfile, hashTable h, int f_or_c);
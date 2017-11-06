---
title: "Suffix Trees"
layout: post
date: 2017-10-20 22:48
headerImage: false
tag:
- Suffix trees
- Python
category: blog
author: rahulmayuranath
description:  Suffix Trees in Python
---


	 	 	
Suffix tree is a compressed trie of all the suffixes of a given string.
Edges that direct to a node having single child are combined together to form a single edge and their edge labels are concatenated. Hence, each internal node has at least two children. The space complexity of the suffix tree is O(n), if optimisations are made.

### Construction of the suffix tree:

The following resource was used a reference while building the suffix tree:
<http://www.cs.jhu.edu/~langmea/resources/lecture_notes/suffix_trees.pdf>

Each node in the suffix tree has :
1. Label on path from this edge to the next node
2. A dictionary of outgoing edges; Characters are mapped to nodes.
3. Leaves; All leaf nodes of the corresponding node.

To insert into the suffix tree, we first insert the longest suffix(i.e the entire string) into the suffix tree. The root node will map the first character with an edge label comprising the entire string.
Now, we keep inserting smaller suffixes until we exhaust the suffixes.
We either fall off in the middle of an edge or fall off at a node.
The latter case is easy to handle since we just have to add an edge hanging off the current node.
To handle the former case, we create a new middle node from where we “fell off” the tree.
The original child becomes the node node’s child and its label is curtailed from where the mismatch occurs. The new child is also added to the current node.
	 	 	
The above algorithm builds a suffix tree for a single string. 
A Generalized Suffix Tree(GST) is constructed for all the string in the document. Once the all the strings are added to the above tree in a similar way, we proceed to insert the leaf nodes. 
We need to keep track of which position the string is in the text; To achieve this, every leaf node consists the position number of the that string in the document. 
Now the substring problem is just reduced to returning text[index] for all indexes present in the leaves.

We should also preprocess the text so that we can segregate tale titles and their corresponding tale. 
The problems can be answered now that we have the suffix tree built and the text has been preprocessed.  


* #### List all the occurrences of a query-string in the set of documents. <br>
We iterate over each of the tales individually and search for the query string. If we end at a leaf node, we return all the indexes contained in the leaf string. If we end at an internal node, we DFS over all the nodes rooted at the current node and return the indexes from all the leaves. This will return all occurrences of the string. Once we get all the indexes, we also print a few words surrounding the word. Getting the corresponding document is trivial since we are iterating for each document separately. This takes O(n+k) for each tale, where k is the number of the matches.
{% highlight python %}
   def dfs(self, node, visited):
        """ For tree rooted at the given node, we recursively visit all children nodes 
           until we exhaust the tree. """
        if(node not in visited):
            visited.append(node)
        for nodes in node.out:
            self.dfs(node.out[nodes], visited)
        return visited
    
   def getLeaves(self, s):
        """ We DFS and get all nodes rooted below the node. We iterate over their
            leaves and get the corresponding positions and return a list of positions"""
        res = []
        node, offset = self.followPath(s)
        visited = self.dfs(node, [])
        for v in visited:
            for a, b in v.leaf.items():
                res.extend(b)
        return res
{% endhighlight %}

* #### List only the first occurrence of a given query-string in every document. If the query-string is not present, find the first occurrence of the longest substring of the query-string. <br>
This question is similar to the one above with minor modifications. 
We do the same as above, but only return the smallest index for each tale that are returned by the leaves. This will trivially be the first occurrence, since it has the smallest index. If the string is not present in the text, we slice the string and keep trying for smaller substrings, until an exact match is found.<br>   
This will take O(n+k)z time for each tale, where z is number of substrings. 


* #### For a given query-string (query of words), list the documents ranked by the relevance.
We maintain a list of ranks for each of the documents.
The ranking criteria is as follows:
For a query string entered, we split it into its corresponding words.
If an exact match is found for a word, we increment the rank of that document by 100. If a match is not found, we try for smaller substrings until we exhaust the string. For each smaller match, the rank reduces. (Eg: Matching banana fetches 100, matching banan fetches 50).
This is done for every word in the query string.
At last, we sort the ranks by their index and return the rank of the documents.
{% highlight python %}
query = "occasion when the shepherd laid hold of him"
# Words is a list consisting of all the words of the query string. 

words = query.split()

# List of ranks to rank the document. 
""" We first look for an exact match of the word. If it is not present, 
    we look for smaller substrings. We assign a score of 100/z for a match 
    that is found, where z is the slice index. Trivially, exact matches will have 
    a higher total score. Finally we sort the rank list by index(Not by magnitude of rank)
    and return the list of documents """

ranks = [0] * 312
for queryWord in words:
    for doc in range(312):
        tree = SuffixTree()
        lenFinal = len(text[doc])
        finalWords = text[doc]
        for i in range(lenFinal):
            found = False
            z = 0
            tree.add(finalWords[i], i)
        for i in range(lenFinal):
            tree.insertLeaves(finalWords[i], i)
        while not found:
            try:
                k = tree.getLeaves(queryWord[z:])
                # We increment total rank of the document and choose the first occurance.  
                found = True
                ranks[doc] += 100 / z
                leaves = min(k)
            except:
                z += 1
                pass


#Sorting ranks by their index
ranks = sorted(range(len(ranks)), key=lambda k: ranks[k])

#Printing documents 
for i in range(len(ranks)):
    print("Rank ", i + 1, " : ", document[ranks[i]])

{% endhighlight %}

The github repo of the entire implentation is [here](https://github.com/rahul-1996/Suffix-Tree). Thanks for reading!





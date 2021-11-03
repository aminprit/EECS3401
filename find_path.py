# main.py
import sys
from queue import PriorityQueue, Queue

# Add a vertex to the dictionary
def add_vertex(v):
  global graph
  global vertices_no
  if v in graph:
    pass
    # print("Vertex ", v, " already exists.")
  else:
    vertices_no = vertices_no + 1
    graph[v] = []

# Add an edge between vertex v1 and v2 with edge weight e
def add_edge(v1, v2, e):
  global graph
  # Check if vertex v1 is a valid vertex
  if v1 not in graph:
    pass
    # print("Vertex ", v1, " does not exist.")
  # Check if vertex v2 is a valid vertex
  elif v2 not in graph:
    pass
  else:
    temp = [v2, e]
    graph[v1].append(temp)
    temp = [v1, e]
    graph[v2].append(temp)

def get_children(vertex):
  children = []
  for child in graph[vertex]:
    children.append(child)

# Print the graph
def print_graph():
  global graph
  global l2
  l2 = []
  for vertex in graph:
    for edges in graph[vertex]:
      # print(vertex, " -> ", edges[0], " edge weight: ", edges[1])
      l2.append((vertex, edges[0], edges[1]))
  # print(l2)

def neighbour(l2, kono):
  l3 = set()
  for x in l2:
    if (kono == x[0]):
      l3.add(x[1])
    if (kono == x[1]):
      l3.add(x[0])
  return l3

# driver code
graph = {}
# stores the number of vertices in the graph
vertices_no = 0

# def uniform_cost_search(source, destination):
#   global path, queue
#   path = []  
#   if source == destination:
#     return path

#   queue = []
#   queue.append(source)

#   solution = "failure"

#   while(len(queue) > 0 and cheaper(solution, queue)):
    
#     parent = queue[0]
#     queue.pop(0)

#     for child in get_children(parent):
#       if not queue._contains_(child):
#         queue[] = 
#         queue.append(child[0])

# def cheaper(solution, vertex_cost):
#   if solution == "failure":
#     return True
#   return (vertex_cost < solution) 


def DFS(graph, start, end, path, shortest):
    path = path + [start] 
    # print('Current DFS path:', path)
    if start == end:
        return path

    x =  neighbour(l2, start)
    for node in x:
        if node not in path: #avoid cycles
            if shortest == None or len(path) < len(shortest):
                newPath = DFS(graph, node, end, path, shortest)
                if newPath != None:
                    shortest = newPath
    return shortest


def bfs(source, destination):
  global bfs_traversal_output
  visited  = {}
  level = {}
  parent = {}
  bfs_traversal_output = []
  queue = Queue()
  
  for node in graph:
    visited[node] = False
    parent[node] = False
    level[node] = -1
  
  visited[source] = True
  level[source] = 0
  queue.put(source)

  while not queue.empty():
    u = queue.get()
    bfs_traversal_output.append(u)

    for v in neighbour(l2, u):
      
      if u == destination:
        return parent
        
      if not visited[v]:
        visited[v] = True
        parent[v] = u
        level[v] = level[u] + 1
        queue.put(v)
  return parent
def h_v(l4, source):
  for i in l4:
    if (i[0] == source):
      return int(i[1])
  return 0
def astar(source, destination, l2):
  
  dist = {}
  previous = {}
  for v in graph:
    dist[v] = float("inf")
    previous[v] = None
  
  dist[source] = 0
  queue = []
  queue.append(source)
  
  while len(queue) != 0:
    u = queue.pop()
    if(u == destination):
      return previous
    for neighbor in neighbour(l2, u):
      cost = dist[u] + int(distance(u, neighbor)) + int(h_v(l4, neighbor) )
      # print(cost)
      # print(u,neighbor)
      # print(distance(u, neighbor))
      if(cost < dist[neighbor] or neighbor not in dist):
        dist[neighbor] = cost
        queue.append(neighbor)
        previous[neighbor] = u

  return previous

def dijkstra(source, destination, l2):
  
  dist = {}
  previous = {}
  for v in graph:
    dist[v] = float("inf")
    previous[v] = None
  
  dist[source] = 0
  queue = []
  queue.append(source)
  
  while len(queue) != 0:
    u = queue.pop()
    if(u == destination):
      return previous
    for neighbor in neighbour(l2, u):
      cost = dist[u] + int(distance(u, neighbor))
      # print(cost)
      # print(u,neighbor)
      # print(distance(u, neighbor))
      if(cost < dist[neighbor] or neighbor not in dist):
        dist[neighbor] = cost
        queue.append(neighbor)
        previous[neighbor] = u

  return previous
 

def print_sol(s, l2):
  sum = 0
  path_sol = ""

  if s == None:
    print("distance: infinity")
    print("path: ")
    print("none")
  else:
    for i in range(len(s) - 1):
        for x in l2:
          if (s[i] == x[0] and s[i + 1] == x[1]) :
            sum += int(x[2])
            path_sol += str(s[i]) + " to " + str(s[i+1]) + ": " + str(x[2]) + " mi\n"

    if(path_sol != None):
      print("distance: " + str(sum) +" mi")
      print("path: ")
      print(path_sol)

def print_sol_bfs(s, l2):
  sum1 = 0
  path_sol1 = ""

  if len(s) == 0:
    print("distance: infinity")
    print("path: ")
    print("none")
  else:
    for i in range(len(s) - 1):

      for x in l2:
          if (s[i] == x[0] ) and (s[i + 1] == x[1] ):
            sum1 += int(x[2])
            path_sol1 += str(s[i]) + " to " + str(s[i+1]) + ": " + str(x[2]) + " mi\n"

    if(path_sol1 != None):
      
      print("distance: " + str(sum1) +" mi")
      print("path: ")
      print(path_sol1)

      


def print_sol_ucs(s, l2):
  sum1 = 0
  path_sol1 = ""

  if len(s) == 0:
    print("distance: infinity")
    print("path: ")
    print("none")
  else:
    for i in range(len(s) - 1):

      for x in l2:
          if (s[i] == x[0] ) and (s[i + 1] == x[1] ):
            sum1 += int(x[2])
            path_sol1 += str(s[i]) + " to " + str(s[i+1]) + ": " + str(x[2]) + " mi\n"

    if(path_sol1 != None):
      print("distance: " + str(sum1) +" mi")
      print("path: ")
      print(path_sol1)

def distance(u, v):
  d = 0
  for i in l2:
    if (i[0] == u and i[1] == v):
      d = i[2]
    if (i[0] == v and i[1] == u):
      d = i[2]
      # if(l2[i][0] == sys.argv[3] and l2[i][1] == sys.argv[4]) or (l2[i][0] == sys.argv[4] and l2[i][1] == sys.argv[3]):
        # d = l2[i][2] 
  return d

if __name__ == "__main__" :
  
    graph = {}
    l1 = []
    f = open(sys.argv[2],'r')
    # print(f.readline())
    x = f.readline()

    r = x.split()
    while(x != "END" and len(x) != x) : 
        # print(r)
        add_vertex(r[0])
        add_vertex(r[1]) 
        add_edge(r[0], r[1], r[2])
        l1.append((r[0], r[1], r[2]))
        x = f.readline()
        r = x.split()
    # print(l1)
    print_graph()
    if(sys.argv[1] == "dfs"):
      p = []    
      s = DFS(l2, sys.argv[3], sys.argv[4], p, None )   
      print_sol(s, l2)
    
    if (sys.argv[1] == "bfs" ):
      s = bfs(sys.argv[3], sys.argv[4])
      # print(s)
      f = []
      c = sys.argv[4]
      if(sys.argv[3] == sys.argv[4]):
        f.append(sys.argv[3])
      else:
        while(c != False):
        
          f.append(c)
          if(s.get(c) == False):
            f.remove(c)
          c = s.get(c)
      if(len(f) != 0):
        f.append(sys.argv[3])

      f.reverse()

      print_sol_bfs(f, l2)

    if(sys.argv[1] == "ucs"):
      s = dijkstra(sys.argv[3], sys.argv[4], l2)
      g = []
      c = sys.argv[4]
      while(c != None):
          
            g.append(c)
            if(s.get(c) == False):
              g.remove(c)
            c = s.get(c)
      g.reverse() 
      print_sol_ucs(g, l2)

    if(sys.argv[1] == "astar"):
      global l4
      l4 =[]
      f = open(sys.argv[5],'r')
      # print(f.readline())
      x = f.readline()

      r = x.split()
      while(x != "END" and len(x) != 0) : 
        # if (len(r) != 0):
        #   l4.append((r[0],r[1]))
        
        x = f.readline()
        
        r = x.rstrip()
        r = x.split()
        
      s = astar(sys.argv[3], sys.argv[4], l2)

      g = []
      c = sys.argv[4]
      while(c != None):
          
            g.append(c)
            if(s.get(c) == False):
              g.remove(c)
            c = s.get(c)
      g.reverse() 
      print_sol_ucs(g, l2)
    # for i in s:
    #   print(s.get(i))
    

    # print(distance(sys.argv[3], sys.argv[4]))

    # for i in range(len(l2)):
    #   print("hi", i)
    #   if(l2[i][0] == sys.argv[3] and l2[i][1] == sys.argv[4]) or (l2[i][0] == sys.argv[4] and l2[i][1] == sys.argv[3]):
    #     print(l2[i][2]) 

    # for i in range(len(l2)):
    #   print(l2[i])
    # print(graph["Toledo"][0][1])
  # https://www.educative.io/edpresso/how-to-implement-a-graph-in-python
  #  https://python-forum.io/thread-18179.html
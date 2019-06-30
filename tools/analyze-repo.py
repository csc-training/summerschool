import requests
import json
from datetime import datetime

# repo info
#r = requests.get('https://api.github.com/repos/csc-training/summerschool')
#if(r.ok):
#    repo = json.loads(r.text or r.content)
#
#    print("Repository created: ", repo['created_at'])
#    print("forks: ", repo['forks'])
#    #for key,val in repo.items():
#    #    print("key {} val {}".format(key,val))
#

#--------------------------------------------------

def get_forks(owner, reponame):

    params={
            'sort':'newest',
            'per_page':100,
            }

    #get forks from github3 API
    r = requests.get(
            'https://api.github.com/repos/{}/{}/forks'.format(owner, reponame),
            params=params,
            )

    if(r.ok):
        forks = json.loads(r.text or r.content)
    
        print("all forks of repo: {}:".format(reponame))
        for fork in forks:
            #print("-------------------------------")
            #for key in fork.keys():
            #    print(key)
            #    print("fork:  {} -> {}".format(key, fork[key]))
    
            print(fork['owner']['login'])
    
        print("number of forks: {}".format( len(forks )))

    #from list to dict
    repos = {}
    for fork in forks:
        key = fork['owner']['login']
        repos[key] = fork

    return repos
    
repos = get_forks('csc-training', 'summerschool')

#--------------------------------------------------
#split only this summer school
json_time_format='%Y-%m-%dT%H:%M:%SZ'

split_time = datetime.strptime('2019-04-01T00:00:00Z', json_time_format)
ss19 = {}

print()
print("2019 participant forks:")
for owner in repos.keys():
    repo = repos[owner]

    #2019-06-26T07:06:20Z
    t0 = datetime.strptime(
            repo['created_at'], json_time_format)

    if t0 > split_time:
        print("owner: {}".format(owner))
        ss19[owner] = repo

#--------------------------------------------------
print("number of participants: {}".format(len(ss19.keys() )))
print()


def get_commits(owner, reponame, since=None):
    params={
            'author':owner,
            'per_page':100,
            }

    print("Fetching commits of user: {}".format(owner))

    if not(since == None):
        params['since'] = since

    #get forks from github3 API
    r = requests.get(
            'https://api.github.com/repos/{}/{}/commits'.format(owner, reponame),
            params=params,
            )

    #dig out only commits
    if(r.ok):
        commits = json.loads(r.text or r.content)

        return commits


#--------------------------------------------------
# testing of previous functions

if True:
    user = 'KerttuK'
    repo = repos[user]

    #for key in repo.keys():
    #    print(key)
    #    print("fork:  {} -> {}".format(key, repo[key]))
    #print(repo['created_at'])
    
    commits = get_commits(user, 'summerschool')
    
    print("-------")
    for commit in commits:
        print(commit['commit']['message'])
    




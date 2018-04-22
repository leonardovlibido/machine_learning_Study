JUPYTER-NOTEBOOK tips/shortcuts
    ctrl+shift+p    == autocomplete for commands
    Esc/enter       == normal/insert mode
    A           == insert Above
    B           == insert Bellow
    DD          == delete current cell
    shift+Tab   == Docstring

numpy:
    np.linspace
    x.reshape
    np.random.rand

    np.vstack([np.ones(n), x_train.ravel()])
    x.transpose
    np.linalg.inv
    np.dot
    np.matmul

    x.shape

    np.linalg.pinv() mur-penrouzov pseudoinverz (XT * X)^-1 * X

pandas
    x = pd.DataFrame(data.data, columns=data.feature_names)
    x.head()
    pd.DataFrame.hist(x, figsize=[16,16])
    x.corr() correlationMatrix
    x.values() - ili tako nesto za ndarray

preprocessing
    sc = preprocessing.StandardScaler()
    sc.fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)

matplotlib:
    plt.plot(x,y,'o')
    plt.imshow(cv_test_score)
    plt.show()


model_selection
    xTr, xTe, yTr, yTe = model_selection.train_test_split(x,y,train_size=0.6, test_size=0.4, random_state=7)

    param_grid = {'C' : [10**i for i in range(-5, 5)]}
    grid = model_selection.GridSearchCV(model, param_grid, scoring='accuracy')



linear_model
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    b1 = model.coef_[0][0]
    b0 = model.intercept_[0]
    model.predict(23)
    probabilities = model.predict_proba(23)    

    y_predicted = model.predict(x_test)
    linear_model.score(x_test, y_test)

svm
    model = svm...

feature_extractions
    vectorizer = feature_extraction.text.CountVectorizer()
    vectorizer = feature_extraction.text.TfidVectorizer()

    vectorizer.fit(x)
    x_vectorized = vectorized.transform(x)
    x_vectorized_arr = x_vectorized.toarray()



metrics
    mse = mean_squared


def modelScore(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    return [train_score, test_score]

knnMod = neighbors.KNeighborsClassifier(n_neighbors=3, metric='cosine')
knn_scores = modelScore(knnMod, x_train, y_train, x_test, y_test)

